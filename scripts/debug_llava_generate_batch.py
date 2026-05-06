import argparse
import time
import traceback

import torch
from PIL import Image


def build_prompt(conv_templates, conv_template: str, question: str) -> str:
    # Minimal prompt builder compatible with llava v1.5 chat template.
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--conv-template", default="vicuna_v1")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        model_name,
        device_map="cuda:0",
        multimodal=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    # Create a deterministic dummy image so we don't depend on dataset/image paths.
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))

    # Build per-sample prompts
    q = DEFAULT_IMAGE_TOKEN + "\n" + "What color is the image? Answer with a single word."
    prompts = [build_prompt(conv_templates, args.conv_template, q) for _ in range(args.batch)]

    # Tokenize + pad
    input_ids_list = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).cuda()
    attention_mask = input_ids.ne(pad_id).cuda()

    # Process images into a batched tensor [B,3,H,W] when possible.
    images = process_images([img] * args.batch, image_processor, model.config)
    if isinstance(images, list):
        images = torch.stack(images, dim=0)
    images = images.to(device="cuda", dtype=torch.float16)

    print(f"input_ids={tuple(input_ids.shape)} images={tuple(images.shape)} attn={args.attn_implementation}")
    t0 = time.time()
    out_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=images,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_id,
    )
    dt = time.time() - t0
    texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    print(f"generate_ok batch={args.batch} out_ids={tuple(out_ids.shape)} time_s={dt:.3f}")
    for i, t in enumerate(texts[: min(len(texts), 3)]):
        print(f"[{i}] {t[-200:]}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

