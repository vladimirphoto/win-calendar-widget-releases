"""Загрузка и сохранение изображений (с сохранением EXIF и ICC-профиля)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # необязательная поддержка RAW
    import rawpy  # type: ignore
except Exception:  # pragma: no cover
    rawpy = None

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
STD_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXT = STD_EXT | (RAW_EXT if rawpy is not None else set())


@dataclass
class Meta:
    exif: Optional[bytes] = None
    icc: Optional[bytes] = None


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXT


def load_image(path: str | Path) -> tuple[np.ndarray, Meta]:
    """Возвращает (BGR uint8, метаданные). Ориентация EXIF уже применена к пикселям."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXT:
        if rawpy is None:
            raise RuntimeError("Для RAW установите пакет rawpy: pip install rawpy")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), Meta()

    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        exif_obj = im.getexif()
        im = ImageOps.exif_transpose(im)
        if len(exif_obj) and 274 in exif_obj:
            exif_obj[274] = 1  # пиксели уже повёрнуты
        exif = exif_obj.tobytes() if len(exif_obj) else None
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), Meta(exif=exif, icc=icc)


def load_thumbnail(path: str | Path, size: int = 96) -> np.ndarray:
    """Быстрая миниатюра (BGR). Для JPEG использует draft-режим — не декодирует весь кадр."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        if rawpy is None:
            raise RuntimeError("rawpy не установлен")
        with rawpy.imread(str(path)) as raw:
            th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                im = Image.open(io.BytesIO(th.data))
                im = ImageOps.exif_transpose(im).convert("RGB")
            else:
                im = Image.fromarray(th.data).convert("RGB")
    else:
        im = Image.open(path)
        try:
            im.draft("RGB", (size * 2, size * 2))
        except Exception:
            pass
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size))
    return cv2.cvtColor(np.asarray(im, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def downscale_to(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    s = long_side / m
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def save_image(path: str | Path, img_bgr: np.ndarray, meta: Optional[Meta] = None,
               quality: int = 95) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    kw: dict = {}
    if meta is not None:
        if meta.icc:
            kw["icc_profile"] = meta.icc
        if meta.exif and ext in {".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".png"}:
            kw["exif"] = meta.exif
    if ext in {".jpg", ".jpeg"}:
        kw.update(quality=int(quality), subsampling=0, optimize=True)
        fmt = "JPEG"
    elif ext == ".png":
        fmt = "PNG"
    elif ext in {".tif", ".tiff"}:
        kw["compression"] = "tiff_lzw"
        fmt = "TIFF"
    elif ext == ".webp":
        kw["quality"] = int(quality)
        fmt = "WEBP"
    else:
        fmt = "BMP"
        kw = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format=fmt, **kw)
