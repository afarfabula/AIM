import os
from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None
GQA_ID2IDX = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    global GQA_ID2IDX
    if GQA_RAW_IMAGE_DATASET is None:
        # NOTE: do NOT force token=True here.
        # In some environments a bad/expired HF token can make the hub treat the repo as "not found",
        # even though the dataset is public. Use anonymous access by default.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or False
        GQA_RAW_IMAGE_DATASET = load_dataset(
            "lmms-lab/GQA",
            "testdev_balanced_images",
            split="testdev",
            token=token,
        )
        GQA_ID2IMAGE = {}
        GQA_ID2IDX = {}

        # Build an id -> row index map without decoding all images.
        # Accessing the full rows would eagerly decode the "image" column and be very slow.
        if "id" in GQA_RAW_IMAGE_DATASET.column_names:
            ids = GQA_RAW_IMAGE_DATASET["id"]
        elif "imageId" in GQA_RAW_IMAGE_DATASET.column_names:
            ids = GQA_RAW_IMAGE_DATASET["imageId"]
        else:
            ids = []
        for i, _id in enumerate(ids):
            if _id not in GQA_ID2IDX:
                GQA_ID2IDX[_id] = i

    image_id = doc.get("imageId") or doc.get("image_id") or doc.get("id")
    if image_id not in GQA_ID2IMAGE:
        idx = GQA_ID2IDX.get(image_id)
        if idx is None:
            raise KeyError(f"GQA image id not found in image dataset: {image_id}")
        GQA_ID2IMAGE[image_id] = GQA_RAW_IMAGE_DATASET[idx]["image"].convert("RGB")
    return [GQA_ID2IMAGE[image_id]]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"
