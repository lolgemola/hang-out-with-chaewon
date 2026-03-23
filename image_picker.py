import os
import random
import time
from PIL import Image
from typing import Optional

REACTIONS_DIR = os.path.join(os.path.dirname(__file__), "reactions")
COOLDOWN_SECONDS = 1.5

LABEL_FOLDERS = {
    "happy": "happy",
    "surprised": "surprised",
    "angry": "angry",
    "neutral": "neutral",
    "thumbs_up": "thumbs_up",
    "peace_sign": "peace_sign",
    "open_palm": "open_palm",
    "fist": "fist",
}

_current_image: Optional[Image.Image] = None
_last_switch: float = 0.0
_last_label: str = ""


def get_image(label: str) -> Optional[Image.Image]:
    global _current_image, _last_switch, _last_label

    now = time.time()
    cooldown_expired = (now - _last_switch) >= COOLDOWN_SECONDS
    label_changed = label != _last_label

    if not label_changed and not cooldown_expired:
        return _current_image

    folder_name = LABEL_FOLDERS.get(label, "default")
    folder_path = os.path.join(REACTIONS_DIR, folder_name)

    images = _get_image_files(folder_path)
    if not images:
        folder_path = os.path.join(REACTIONS_DIR, "default")
        images = _get_image_files(folder_path)

    if not images:
        return None

    path = random.choice(images)
    _current_image = Image.open(path).convert("RGBA")
    _last_switch = now
    _last_label = label
    return _current_image


def _get_image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
