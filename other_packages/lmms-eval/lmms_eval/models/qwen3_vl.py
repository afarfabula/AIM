import base64
import json
import os
import urllib.request
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _choose_qwen3_model_cls(pretrained: str):
    config = AutoConfig.from_pretrained(pretrained)
    architectures = set(getattr(config, "architectures", []) or [])
    model_type = getattr(config, "model_type", None)
    if "Qwen3VLMoeForConditionalGeneration" in architectures or model_type == "qwen3_vl_moe":
        return Qwen3VLMoeForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prefetch_qwen3_snapshot(pretrained: str) -> str:
    if os.path.isdir(pretrained):
        return pretrained

    cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME")
    endpoint = os.getenv("HF_ENDPOINT")
    max_workers = int(os.getenv("QWEN3_MODEL_SNAPSHOT_MAX_WORKERS", "1"))
    force_download = _env_flag("QWEN3_MODEL_FORCE_DOWNLOAD", False)
    resume_download = not _env_flag("QWEN3_MODEL_DISABLE_RESUME", False)

    try:
        return snapshot_download(
            repo_id=pretrained,
            cache_dir=cache_dir,
            endpoint=endpoint,
            resume_download=resume_download,
            force_download=force_download,
            max_workers=max_workers,
            etag_timeout=60,
        )
    except Exception as e:
        if "Consistency check failed" not in str(e):
            raise
        eval_logger.warning(
            "Qwen3 model snapshot download hit consistency check failure; retrying with force_download=True, "
            "resume_download=False, max_workers=1"
        )
        return snapshot_download(
            repo_id=pretrained,
            cache_dir=cache_dir,
            endpoint=endpoint,
            resume_download=False,
            force_download=True,
            max_workers=1,
            etag_timeout=60,
        )


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3-VL model wrapper for lmms-eval.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        dtype: Optional[str] = None,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 3136,
        max_num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # lmms-eval may inject internal bookkeeping kwargs.
        kwargs.pop("virtual_world_size", None)
        kwargs.pop("virtual_rank", None)
        if kwargs:
            eval_logger.warning(f"Ignoring unsupported qwen3_vl kwargs: {kwargs}")

        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for qwen3_vl. Install it via `pip install qwen-vl-utils`.")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if attn_implementation is None and use_flash_attention_2:
            attn_implementation = "flash_attention_2"

        resolved_pretrained = _prefetch_qwen3_snapshot(pretrained)
        model_cls = _choose_qwen3_model_cls(resolved_pretrained)
        # Qwen3-VL is published in bf16, and `dtype=auto` was enough to trigger
        # SIGFPE during greedy generation on this worker for both FA2 and SDPA.
        if dtype is None and self._device.type == "cuda":
            dtype = "bfloat16"
        model_kwargs = {
            "device_map": self.device_map,
        }
        if dtype is None or dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["dtype"] = getattr(torch, dtype)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = model_cls.from_pretrained(resolved_pretrained, **model_kwargs).eval()

        self.processor = AutoProcessor.from_pretrained(resolved_pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(resolved_pretrained)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)
        self._config = self.model.config

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    @staticmethod
    def _normalize_visuals(visual_entry):
        if visual_entry is None:
            return None
        if isinstance(visual_entry, (list, tuple)):
            if len(visual_entry) == 0:
                return None
            if len(visual_entry) == 1:
                return visual_entry[0]
            return list(visual_entry)
        return visual_entry

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        rgb_image = image.convert("RGB")
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [self._normalize_visuals(doc_to_visual[0](self.task_dict[task][split][ids])) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0].copy()
            until = [self.tokenizer.decode(self.eot_token_id)]
            # #region debug-point A:batch-entry
            exec("try:\n _p='/mlx_devbox/users/quyanyi/playground/AIM/.dbg/qwen3-gqa-fpe.env'; _u,_s='http://127.0.0.1:7777/event','qwen3-gqa-fpe';\n with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'A','location':'qwen3_vl.py:242','msg':'[DEBUG] qwen3 generate batch entry','data':{'task':task,'split':split,'batch_size':len(contexts),'visual_types':[type(v).__name__ for v in visuals],'gen_kwargs_raw':{k:repr(v) for k,v in gen_kwargs.items()}}}).encode(),headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept Exception:\n pass")
            # #endregion

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            contexts = list(contexts)
            messages = []
            for context, visual in zip(contexts, visuals):
                clean_context = context.replace("<image>", "").replace("<video>", "")
                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    vr = decord.VideoReader(visual)
                    first_frame = vr[0].asnumpy()
                    height, width = first_frame.shape[:2]
                    max_pixels = min(self.max_pixels, height * width)
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": max_pixels,
                                    "nframes": self.max_num_frames,
                                },
                                {"type": "text", "text": clean_context},
                            ],
                        }
                    )
                elif isinstance(visual, Image.Image):
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": self._image_to_data_url(visual)},
                                {"type": "text", "text": clean_context},
                            ],
                        }
                    )
                elif isinstance(visual, list) and all(isinstance(v, Image.Image) for v in visual):
                    image_content = [{"type": "image", "image": self._image_to_data_url(v)} for v in visual]
                    message.append({"role": "user", "content": image_content + [{"type": "text", "text": clean_context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": clean_context}]})

                messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs is not None:
                for idx, video in enumerate(video_inputs):
                    total_frames = video.shape[0]
                    if total_frames > self.max_num_frames:
                        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                        if total_frames - 1 not in indices:
                            indices = np.append(indices, total_frames - 1)
                        video_inputs[idx] = video[indices]

            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            # #region debug-point C:inputs-ready
            exec("try:\n _p='/mlx_devbox/users/quyanyi/playground/AIM/.dbg/qwen3-gqa-fpe.env'; _u,_s='http://127.0.0.1:7777/event','qwen3-gqa-fpe';\n with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n _data={};\n _ids=getattr(inputs,'input_ids',None); _mask=getattr(inputs,'attention_mask',None); _pix=getattr(inputs,'pixel_values',None); _grid=getattr(inputs,'image_grid_thw',None);\n _data.update({'input_ids_shape':list(_ids.shape) if _ids is not None else None,'attention_mask_shape':list(_mask.shape) if _mask is not None else None,'pixel_values_shape':list(_pix.shape) if _pix is not None else None,'pixel_values_dtype':str(_pix.dtype) if _pix is not None else None,'pixel_values_device':str(_pix.device) if _pix is not None else None,'image_grid_thw':_grid.tolist() if _grid is not None else None,'image_inputs_count':len(image_inputs) if image_inputs is not None else 0,'video_inputs_count':len(video_inputs) if video_inputs is not None else 0});\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'C','location':'qwen3_vl.py:310','msg':'[DEBUG] qwen3 inputs ready','data':_data}).encode(),headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept Exception:\n pass")
            # #endregion

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # #region debug-point B:before-generate
            exec("try:\n _p='/mlx_devbox/users/quyanyi/playground/AIM/.dbg/qwen3-gqa-fpe.env'; _u,_s='http://127.0.0.1:7777/event','qwen3-gqa-fpe';\n with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'B','location':'qwen3_vl.py:320','msg':'[DEBUG] qwen3 before generate','data':{'do_sample':bool(gen_kwargs['temperature'] > 0),'temperature':repr(gen_kwargs['temperature']),'top_p':repr(gen_kwargs['top_p']),'num_beams':repr(gen_kwargs['num_beams']),'max_new_tokens':repr(gen_kwargs['max_new_tokens']),'model_dtype':str(getattr(self.model,'dtype',None)),'model_device':str(self.device),'use_cache':self.use_cache}}).encode(),headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept Exception:\n pass")
            # #endregion

            generate_kwargs = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": bool(gen_kwargs["temperature"] > 0),
                "num_beams": gen_kwargs["num_beams"],
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }
            if generate_kwargs["do_sample"]:
                generate_kwargs["temperature"] = gen_kwargs["temperature"]
                generate_kwargs["top_p"] = gen_kwargs["top_p"]
            cont = self.model.generate(**inputs, **generate_kwargs)
            # #region debug-point D:after-generate
            exec("try:\n _p='/mlx_devbox/users/quyanyi/playground/AIM/.dbg/qwen3-gqa-fpe.env'; _u,_s='http://127.0.0.1:7777/event','qwen3-gqa-fpe';\n with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in _c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\n urllib.request.urlopen(urllib.request.Request(_u,data=json.dumps({'sessionId':_s,'runId':'pre-fix','hypothesisId':'D','location':'qwen3_vl.py:332','msg':'[DEBUG] qwen3 after generate','data':{'cont_shape':list(cont.shape) if hasattr(cont,'shape') else None}}).encode(),headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept Exception:\n pass")
            # #endregion

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for idx, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[idx] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
