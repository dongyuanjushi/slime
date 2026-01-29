import base64
import ast
from typing import Union
import math
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def encode_image(image: Union[bytes, str]) -> str:
    if isinstance(image, bytes):
        base_64_str = base64.b64encode(image).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    elif isinstance(image, str):
        bytes_obj = bytes_literal_to_bytesio(image)
        base_64_str = base64.b64encode(bytes_obj).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    else:
        raise ValueError("type of screenshot is not supported, only bytes or str is supported")

def bytes_literal_to_bytesio(bytes_literal_str):
    bytes_obj = ast.literal_eval(bytes_literal_str)

    if not isinstance(bytes_obj, bytes):
        raise ValueError("not a valid bytes literal")

    return bytes_obj

def bytes_literal_to_bytesio(bytes_literal_str):
    bytes_obj = ast.literal_eval(bytes_literal_str)

    if not isinstance(bytes_obj, bytes):
        raise ValueError("not a valid bytes literal")

    return bytes_obj

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def encode_screenshot(screenshot: Union[bytes, str]) -> str:
    if isinstance(screenshot, bytes):
        base_64_str = base64.b64encode(screenshot).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    elif isinstance(screenshot, str):
        bytes_obj = bytes_literal_to_bytesio(screenshot)
        base_64_str = base64.b64encode(bytes_obj).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    else:
        raise ValueError("type of screenshot is not supported, only bytes or str is supported")

def process_qwen_image_with_smart_resize(image_bytes, factor=32, min_pixels=1000, max_pixels=1000000000):
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return processed_bytes