import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        token_prune_strategy: str = "none",
        token_prune_config: Optional[str] = None,
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        virtual_world_size: int = 1,
        virtual_rank: int = 0,
        virtual_device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        virtual_world_size = int(virtual_world_size or 1)
        virtual_rank = int(virtual_rank or 0)
        if virtual_world_size < 1:
            raise ValueError(f"virtual_world_size must be >= 1, got {virtual_world_size}")
        if virtual_rank < 0 or virtual_rank >= virtual_world_size:
            raise ValueError(f"virtual_rank must be in [0, {virtual_world_size}), got {virtual_rank}")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.virtual_data_parallel = virtual_world_size > 1
        self.uses_distributed_gather = False
        if self.virtual_data_parallel:
            target_device = virtual_device or device or "cuda:0"
            self._device = torch.device(target_device)
            self.device_map = target_device
        elif accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self.token_prune_strategy = token_prune_strategy
        self.token_prune_strategy_config = None
        if token_prune_config is not None:
            try:
                import json

                self.token_prune_strategy_config = json.loads(token_prune_config)
                eval_logger.info(f"Parsed token prune config: {self.token_prune_strategy_config}")
            except json.JSONDecodeError as e:
                eval_logger.warning(f"Failed to parse token_prune_config: {e}")

        if self.token_prune_strategy and self.token_prune_strategy != "none":
            try:
                from llava.model.token_prune_strategies import apply_strategy_by_name, list_strategies

                eval_logger.info(f"Applying token prune strategy: {self.token_prune_strategy}")
                self._model = apply_strategy_by_name(self._model, self.token_prune_strategy, self.token_prune_strategy_config)
                eval_logger.info(f"Token prune strategy applied. Available strategies: {list_strategies()}")
            except Exception as e:
                eval_logger.error(f"Failed to apply token prune strategy {self.token_prune_strategy}: {e}")
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()


        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if self.virtual_data_parallel:
            eval_logger.info(f"Using virtual data parallelism on {self._device}: rank {virtual_rank}/{virtual_world_size}")
            self.model.to(self._device)
            self._rank = virtual_rank
            self._world_size = virtual_world_size
        elif accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self.uses_distributed_gather = True
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        debug_batch_prints = 0

        request_args = [reg.args for reg in requests]
        chunks = [request_args[i : i + self.batch_size] for i in range(0, len(request_args), self.batch_size)]
        pbar = tqdm(total=len(chunks), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            gen_kwargs = dict(all_gen_kwargs[0]) if isinstance(all_gen_kwargs[0], dict) else all_gen_kwargs[0]

            if isinstance(gen_kwargs, dict) and "image_aspect_ratio" in gen_kwargs and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(flattened_visuals) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    question = " ".join(image_tokens) + "\n" + context
                else:
                    question = context
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                question_input.append(conv.get_prompt())

            if isinstance(gen_kwargs, dict):
                gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
                gen_kwargs.setdefault("max_new_tokens", 1024)
                gen_kwargs.setdefault("temperature", 0)
                gen_kwargs.setdefault("top_p", None)
                gen_kwargs.setdefault("num_beams", 1)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            cont = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                image_sizes=gen_kwargs["image_sizes"] if isinstance(gen_kwargs, dict) else [],
                do_sample=True if (isinstance(gen_kwargs, dict) and gen_kwargs.get("temperature", 0) > 0) else False,
                temperature=gen_kwargs.get("temperature", 0) if isinstance(gen_kwargs, dict) else 0,
                top_p=gen_kwargs.get("top_p", None) if isinstance(gen_kwargs, dict) else None,
                num_beams=gen_kwargs.get("num_beams", 1) if isinstance(gen_kwargs, dict) else 1,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024) if isinstance(gen_kwargs, dict) else 1024,
                use_cache=self.use_cache,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            if isinstance(text_outputs, str):
                text_outputs = [text_outputs]
            if len(text_outputs) != len(chunk):
                eval_logger.warning(
                    f"decoded_outputs={len(text_outputs)} != chunk_size={len(chunk)}; padding/truncating to match."
                )
                if len(text_outputs) < len(chunk):
                    text_outputs = list(text_outputs) + [""] * (len(chunk) - len(text_outputs))
                else:
                    text_outputs = list(text_outputs)[: len(chunk)]

            if self.batch_size > 1 and debug_batch_prints < 3:
                try:
                    cont_shape = tuple(cont.shape)
                except Exception:
                    cont_shape = "unknown"
                eval_logger.info(
                    f"[batch-debug] chunk_size={len(chunk)} input_ids_shape={tuple(input_ids.shape)} "
                    f"cont_shape={cont_shape} decoded={len(text_outputs)}"
                )
                debug_batch_prints += 1

            res.extend(text_outputs)
            try:
                self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), text_outputs)
            except Exception:
                pass
            pbar.update(1)

        pbar.close()
        # Defensive normalization: evaluator expects one output per request.
        if self.rank == 0 and len(res) != len(requests):
            eval_logger.warning(
                f"generate_until returned {len(res)} outputs for {len(requests)} requests; padding/truncating to match."
            )
        if len(res) < len(requests):
            res = list(res) + [""] * (len(requests) - len(res))
        elif len(res) > len(requests):
            res = list(res)[: len(requests)]
        return res


    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
