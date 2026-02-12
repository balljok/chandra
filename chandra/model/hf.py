import time
from typing import List

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit, detect_repeat_token
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    max_retries: int = None,
    max_failure_retries: int = None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:
    from qwen_vl_utils import process_vision_info

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    def _generate(item: BatchInputItem, temperature: float = 0.6, top_p: float = 0.9) -> GenerationResult:
        message = process_batch_element(item, model.processor, bbox_scale)
        text = model.processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info([message])
        inputs = model.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        inputs = inputs.to("cuda")

        try:
            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=top_p,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = model.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result = GenerationResult(
                raw=output_text[0], token_count=len(generated_ids_trimmed[0]), error=False
            )
        except Exception as e:
            print(f"Error during HF generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return result

    def _should_retry(result, retries, max_retries, max_failure_retries):
        has_repeat = detect_repeat_token(result.raw) or (
            len(result.raw) > 50 and detect_repeat_token(result.raw, cut_from_end=50)
        )

        if retries < max_retries and has_repeat:
            print(
                f"Detected repeat token, retrying generation (attempt {retries + 1})..."
            )
            return True

        if retries < max_retries and result.error:
            print(
                f"Detected hf error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        if (
            result.error
            and max_failure_retries is not None
            and retries < max_failure_retries
        ):
            print(
                f"Detected hf error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        return False

    def process_item(item):
        result = _generate(item)
        retries = 0

        while _should_retry(result, retries, max_retries, max_failure_retries):
            result = _generate(item, temperature=0.7, top_p=0.95)
            retries += 1

        return result

    # Process each item individually with retry logic
    results = [process_item(item) for item in batch]
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
