from typing import List

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import (
    scale_to_fit,
    detect_repeat_token,
    detect_repeat_token_advanced,
)
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    max_retries: int | None = None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:
    from qwen_vl_utils import process_vision_info

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    messages = [
        process_batch_element(item, model.processor, bbox_scale) for item in batch
    ]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = model.processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_output_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]

    for idx, result in enumerate(results):
        retries = 0
        has_repeat = detect_repeat_token_advanced(result.raw) or (
            len(result.raw) > 50
            and detect_repeat_token_advanced(result.raw, cut_from_end=50)
        )

        while retries < max_retries and has_repeat:
            print(
                f"Detected repeat token, retrying generation (attempt {retries + 1})..."
            )

            retry_message = process_batch_element(
                batch[idx], model.processor, bbox_scale
            )
            retry_text = model.processor.apply_chat_template(
                [retry_message], tokenize=False, add_generation_prompt=True
            )
            retry_image_inputs, _ = process_vision_info([retry_message])
            retry_inputs = model.processor(
                text=retry_text,
                images=retry_image_inputs,
                padding=True,
                return_tensors="pt",
                padding_side="left",
            )
            retry_inputs = retry_inputs.to("cuda")

            # Adjusted according to suggestion på ChatGPT
            retry_generated_ids = model.generate(
                **retry_inputs,
                max_new_tokens=max_output_tokens,
                temperature=min(0.65 + 0.08 * (retries + 1), 0.9),
                top_p=min(0.85 + 0.02 * (retries + 1), 0.92),
                # rep_penalty=min(1.05 + 0.03 * (retries + 1), 1.15),
                do_sample=True,
            )
            # retry_generated_ids = model.generate(
            #     **retry_inputs,
            #     max_new_tokens=max_output_tokens,
            #     temperature=0.3,
            #     top_p=0.95,
            #     do_sample=True,
            # )
            retry_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(retry_inputs.input_ids, retry_generated_ids)
            ]
            retry_text_out = model.processor.batch_decode(
                retry_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            results[idx] = GenerationResult(
                raw=retry_text_out, token_count=len(retry_trimmed[0]), error=False
            )

            retries += 1
            has_repeat = detect_repeat_token_advanced(results[idx].raw) or (
                len(results[idx].raw) > 50
                and detect_repeat_token_advanced(results[idx].raw, cut_from_end=50)
            )

    return results


def process_batch_element(item: BatchInputItem, processor, bbox_scale: int):
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        prompt = PROMPT_MAPPING[prompt_type].replace("{bbox_scale}", str(bbox_scale))

    content = []
    image = scale_to_fit(item.image)  # Guarantee max size
    content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})
    message = {"role": "user", "content": content}
    return message


def load_model():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    device_map = "auto"
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings.MODEL_CHECKPOINT, **kwargs
    )
    model = model.eval()
    processor = Qwen3VLProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model.processor = processor
    return model
