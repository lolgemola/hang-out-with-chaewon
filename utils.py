import cv2
import numpy as np
from PIL import Image
from typing import Optional

MAX_W = 240
MAX_H = 240
PADDING = 10


def resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def overlay(
    frame: np.ndarray,
    reaction_img: Optional[Image.Image],
    position: str = "bottom-right",
    alpha: float = 1.0,
) -> np.ndarray:
    if reaction_img is None:
        return frame

    img = resize_to_fit(reaction_img.copy(), MAX_W, MAX_H)
    img_w, img_h = img.size

    fh, fw = frame.shape[:2]

    if position == "bottom-right":
        x = fw - img_w - PADDING
        y = fh - img_h - PADDING
    elif position == "bottom-left":
        x, y = PADDING, fh - img_h - PADDING
    elif position == "top-right":
        x, y = fw - img_w - PADDING, PADDING
    else:
        x, y = PADDING, PADDING

    # convert reaction image to RGBA numpy array
    img_array = np.array(img.convert("RGBA"))
    img_rgb = img_array[:, :, :3][:, :, ::-1]  # RGB -> BGR
    img_alpha = (img_array[:, :, 3] / 255.0) * alpha

    # region of interest on the frame
    roi = frame[y:y + img_h, x:x + img_w]

    # blend
    for c in range(3):
        roi[:, :, c] = (img_alpha * img_rgb[:, :, c] + (1 - img_alpha) * roi[:, :, c]).astype(np.uint8)

    frame[y:y + img_h, x:x + img_w] = roi
    return frame
