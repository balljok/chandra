import math
import re
from typing import Tuple

from PIL import Image

from chandra.output import parse_markdown


def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (28, 28),
):
    resample_method = Image.Resampling.LANCZOS

    width, height = img.size

    # Check for empty or invalid image
    if width == 0 or height == 0:
        return img

    max_width, max_height = max_size
    min_width, min_height = min_size

    current_pixels = width * height
    max_pixels = max_width * max_height
    min_pixels = min_width * min_height

    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5

        new_width = math.floor(width * scale_factor)
        new_height = math.floor(height * scale_factor)
    elif current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5

        new_width = math.ceil(width * scale_factor)
        new_height = math.ceil(height * scale_factor)
    else:
        return img

    return img.resize((new_width, new_height), resample=resample_method)


def is_markup_sequence(text: str, markup_threshold: float = 0.4) -> bool:
    """
    Check if a text sequence is primarily markup (HTML/XML tags).

    Args:
        text: The text sequence to check
        markup_threshold: Minimum ratio of markup characters to consider it markup (default: 0.4)

    Returns:
        True if the sequence appears to be primarily markup
    """
    if not text:
        return False

    # Count angle brackets and common tag patterns
    markup_chars = text.count("<") + text.count(">")

    # Check for common HTML/XML tag patterns
    tag_pattern = r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>"
    tags = re.findall(tag_pattern, text)

    if tags:
        # Calculate the proportion of text that is tags
        tag_length = sum(len(tag) for tag in tags)
        tag_ratio = tag_length / len(text)

        # If tags make up a significant portion, consider it markup
        if tag_ratio >= markup_threshold:
            return True

    # Alternative check: high density of angle brackets
    markup_ratio = markup_chars / len(text)
    return markup_ratio >= markup_threshold


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
):
    try:
        predicted_tokens = parse_markdown(predicted_tokens)
    except Exception as e:
        print(f"Error parsing markdown: {e}")
        return True

    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]

    for seq_len in range(1, window_size // 2 + 1):
        # Extract the potential repeating sequence from the end
        candidate_seq = predicted_tokens[-seq_len:]

        # Skip if this is primarily a markup sequence (natural repetition in HTML/XML)
        if is_markup_sequence(candidate_seq):
            continue

        # Inverse scaling: shorter sequences need more repeats
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))

        # Count how many times this sequence appears consecutively at the end
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue

        while pos >= 0:
            if predicted_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break

        if repeat_count > max_repeats:
            return True

    return False
