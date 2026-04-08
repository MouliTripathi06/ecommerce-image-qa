"""
Synthetic product image generator.
Generates base product images and applies known defects so ground truth is always available.
"""
import random
import io
import base64
from typing import List, Tuple, Dict
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

# All possible defect types
DEFECT_TYPES = ["blur", "dark", "watermark", "wrong_background", "overexposed"]

# Backgrounds considered "correct" (white/light grey)
CORRECT_BACKGROUNDS = ["white", "light_grey"]
WRONG_BACKGROUNDS = ["cluttered", "colored", "dark_bg"]


def generate_base_product_image(size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a clean synthetic product image (white background, simple shape)."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw a simple "product" — a colored rectangle with a label
    colors = [(220, 80, 80), (80, 120, 220), (80, 200, 120), (200, 160, 40)]
    color = random.choice(colors)
    margin = size[0] // 6
    draw.rectangle(
        [margin, margin, size[0] - margin, size[1] - margin],
        fill=color,
        outline=(50, 50, 50),
        width=2,
    )
    # Simple highlight
    draw.ellipse(
        [margin + 10, margin + 10, margin + 40, margin + 40],
        fill=(255, 255, 255, 128),
    )
    return img


def apply_blur(img: Image.Image, radius: float = None) -> Tuple[Image.Image, float]:
    radius = radius if radius is not None else random.uniform(3.0, 8.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), radius


def apply_dark(img: Image.Image, factor: float = None) -> Tuple[Image.Image, float]:
    factor = factor if factor is not None else random.uniform(0.15, 0.45)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor), factor


def apply_watermark(img: Image.Image) -> Image.Image:
    watermarked = img.copy().convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    text = "SAMPLE"
    # Diagonal watermark
    draw.text((img.size[0] // 4, img.size[1] // 3), text, fill=(180, 180, 180, 120))
    draw.text((img.size[0] // 4 + 1, img.size[1] // 3 + 30), "© StockPhoto", fill=(180, 180, 180, 100))
    combined = Image.alpha_composite(watermarked, txt_layer)
    return combined.convert("RGB")


def apply_wrong_background(img: Image.Image, bg_type: str = None) -> Tuple[Image.Image, str]:
    if bg_type is None:
        bg_type = random.choice(WRONG_BACKGROUNDS)
    size = img.size
    if bg_type == "cluttered":
        bg = Image.new("RGB", size, (200, 180, 160))
        draw = ImageDraw.Draw(bg)
        for _ in range(15):
            x, y = random.randint(0, size[0]), random.randint(0, size[1])
            draw.rectangle([x, y, x + 20, y + 20], fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
    elif bg_type == "colored":
        bg = Image.new("RGB", size, (50, 100, 200))
    else:  # dark_bg
        bg = Image.new("RGB", size, (30, 30, 30))

    # Paste product shape on wrong background
    margin = size[0] // 6
    product_region = img.crop((margin, margin, size[0] - margin, size[1] - margin))
    bg.paste(product_region, (margin, margin))
    return bg, bg_type


def apply_overexposed(img: Image.Image, factor: float = None) -> Tuple[Image.Image, float]:
    factor = factor if factor is not None else random.uniform(2.5, 4.0)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor), factor


def generate_image_with_defects(
    defects: List[str] = None,
    size: Tuple[int, int] = (256, 256),
) -> Tuple[str, List[str], Dict]:
    """
    Generate a product image with specified (or random) defects applied.

    Returns:
        (base64_image_str, applied_defects, defect_metadata)
    """
    if defects is None:
        # Random: 0–3 defects, weighted toward 1
        num = random.choices([0, 1, 2, 3], weights=[15, 50, 25, 10])[0]
        defects = random.sample(DEFECT_TYPES, k=num)

    img = generate_base_product_image(size)
    metadata = {"defects_applied": defects, "params": {}}

    for defect in defects:
        if defect == "blur":
            img, radius = apply_blur(img)
            metadata["params"]["blur_radius"] = round(radius, 2)
        elif defect == "dark":
            img, factor = apply_dark(img)
            metadata["params"]["brightness_factor"] = round(factor, 2)
        elif defect == "watermark":
            img = apply_watermark(img)
        elif defect == "wrong_background":
            img, bg_type = apply_wrong_background(img)
            metadata["params"]["bg_type"] = bg_type
        elif defect == "overexposed":
            img, factor = apply_overexposed(img)
            metadata["params"]["overexposed_factor"] = round(factor, 2)

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64, defects, metadata


def compute_severity(defects: List[str], params: Dict) -> float:
    """
    Compute a ground-truth severity score (0.0–1.0) based on defects and their parameters.
    """
    if not defects:
        return 0.0
    score = 0.0
    count = len(defects)

    if "blur" in defects:
        radius = params.get("blur_radius", 5.0)
        score += min(radius / 10.0, 1.0)
    if "dark" in defects:
        factor = params.get("brightness_factor", 0.3)
        score += 1.0 - factor  # darker = more severe
    if "watermark" in defects:
        score += 0.7
    if "wrong_background" in defects:
        score += 0.8
    if "overexposed" in defects:
        factor = params.get("overexposed_factor", 3.0)
        score += min((factor - 1.0) / 3.0, 1.0)

    return round(min(score / max(count, 1), 1.0), 3)


def compute_recommendation(defects: List[str], severity: float) -> str:
    """Ground-truth recommendation based on defects."""
    if not defects:
        return "approve"
    if "watermark" in defects or "wrong_background" in defects or severity > 0.7:
        return "reject"
    return "retouch"
