"""Движок обработки Lumina Retouch.

Чистый numpy + OpenCV, без GUI — поэтому его можно тестировать отдельно и
запускать пакетно. Все радиусы считаются относительно размера кадра, поэтому
превью (уменьшенное) и экспорт (полный размер) выглядят одинаково.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Нормализованный бокс лица: (x, y, w, h) в долях от размера кадра.
FaceBox = Tuple[float, float, float, float]


@dataclass
class Params:
    # --- Кожа ---
    skin_smooth: float = 0.0      # 0..100 сглаживание тона
    skin_texture: float = 70.0    # 0..100 сколько текстуры (пор) сохранить
    blemish: float = 0.0          # 0..100 чувствительность удаления дефектов
    even_tone: float = 0.0        # 0..100 выравнивание покраснений/пятен
    faces_only: bool = True       # ретушировать только область людей (если лица найдены)
    # --- Свет ---
    exposure: float = 0.0         # -2..2 EV
    contrast: float = 0.0         # -100..100
    highlights: float = 0.0       # -100..100
    shadows: float = 0.0          # -100..100
    # --- Цвет ---
    temperature: float = 0.0      # -100..100 (минус — холоднее)
    tint: float = 0.0             # -100..100 (плюс — к пурпурному)
    saturation: float = 0.0       # -100..100
    vibrance: float = 0.0         # -100..100
    # --- Детали / эффекты ---
    clarity: float = 0.0          # -100..100 локальный контраст
    sharpen: float = 0.0          # 0..100
    vignette: float = 0.0         # -100..100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def copy(self) -> "Params":
        return Params.from_dict(self.to_dict())

    def needs_skin_mask(self) -> bool:
        return self.skin_smooth > 0 or self.blemish > 0 or self.even_tone > 0


PRESETS = {
    "Без изменений": Params(),
    "Мягкая ретушь": Params(skin_smooth=35, skin_texture=75, blemish=40, even_tone=25,
                            sharpen=15),
    "Бьюти": Params(skin_smooth=60, skin_texture=55, blemish=65, even_tone=50,
                    exposure=0.15, highlights=-15, shadows=15, vibrance=15,
                    clarity=10, sharpen=25),
    "Тёплый портрет": Params(skin_smooth=30, skin_texture=75, blemish=35, even_tone=25,
                             temperature=22, tint=4, contrast=10, vibrance=18,
                             vignette=15, sharpen=20),
    "Чёрно-белый": Params(skin_smooth=25, skin_texture=80, saturation=-100,
                          contrast=22, clarity=20, vignette=20, sharpen=20),
}


# ----------------------------------------------------------------------------
# Вспомогательное
# ----------------------------------------------------------------------------
def _odd(n: float, lo: int = 3) -> int:
    n = int(round(n))
    n = max(lo, n)
    return n if n % 2 == 1 else n + 1


def _luma(img: np.ndarray) -> np.ndarray:
    # img BGR float32 (быстрый путь через OpenCV, те же коэффициенты Rec.601)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_cascade = None
_cascade_failed = False


def detect_faces(img_bgr: np.ndarray) -> List[FaceBox]:
    """Находит лица (каскады Хаара). Возвращает нормализованные боксы."""
    global _cascade, _cascade_failed
    if _cascade_failed:
        return []
    if _cascade is None:
        try:  # в OpenCV 5.x каскады вынесены из основного пакета — тогда работаем без лиц
            _cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if _cascade.empty():
                raise RuntimeError("cascade not loaded")
        except Exception:
            _cascade_failed = True
            return []
    h, w = img_bgr.shape[:2]
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    sh, sw = gray.shape[:2]
    min_side = max(24, int(min(sh, sw) * 0.06))
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(min_side, min_side))
    out: List[FaceBox] = []
    for (x, y, fw, fh) in faces:
        out.append((x / sw, y / sh, fw / sw, fh / sh))
    return out


# ----------------------------------------------------------------------------
# Маска кожи
# ----------------------------------------------------------------------------
def skin_mask(img_u8: np.ndarray, faces: Sequence[FaceBox], faces_only: bool) -> np.ndarray:
    """Мягкая маска кожи float32 0..1. Если лица найдены — подстраивается под их тон."""
    h, w = img_u8.shape[:2]
    short = min(h, w)
    ycc = cv2.cvtColor(img_u8, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

    cr_lo, cr_hi, cb_lo, cb_hi = 133, 180, 77, 130
    if faces:
        crs, cbs = [], []
        for (fx, fy, fw, fh) in faces:
            x0, x1 = int((fx + fw * 0.25) * w), int((fx + fw * 0.75) * w)
            y0, y1 = int((fy + fh * 0.35) * h), int((fy + fh * 0.85) * h)
            if x1 - x0 > 2 and y1 - y0 > 2:
                crs.append(Cr[y0:y1, x0:x1].ravel())
                cbs.append(Cb[y0:y1, x0:x1].ravel())
        if crs:
            cr_v, cb_v = np.concatenate(crs), np.concatenate(cbs)
            cr_m, cb_m = float(np.median(cr_v)), float(np.median(cb_v))
            cr_t = max(9.0, 2.6 * float(np.std(cr_v)))
            cb_t = max(9.0, 2.6 * float(np.std(cb_v)))
            cr_lo, cr_hi = cr_m - cr_t, cr_m + cr_t
            cb_lo, cb_hi = cb_m - cb_t, cb_m + cb_t

    m = ((Cr >= cr_lo) & (Cr <= cr_hi) & (Cb >= cb_lo) & (Cb <= cb_hi)
         & (Y > 35) & (Y < 250)).astype(np.uint8) * 255
    k = _odd(short * 0.004)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k2 = _odd(k * 2 + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    mf = m.astype(np.float32) / 255.0

    if faces_only and faces:
        roi = np.zeros((h, w), np.float32)
        for (fx, fy, fw, fh) in faces:
            x0 = int(max(0.0, fx - 0.7 * fw) * w)
            x1 = int(min(1.0, fx + 1.7 * fw) * w)
            y0 = int(max(0.0, fy - 0.35 * fh) * h)
            y1 = int(min(1.0, fy + 3.8 * fh) * h)
            cv2.rectangle(roi, (x0, y0), (x1, y1), 1.0, -1)
        sig = max(2.0, max(fw for _, _, fw, _ in faces) * w * 0.25)
        f = max(1.0, sig / 6.0)
        if f > 1.0:  # размываем уменьшенную копию, затем возвращаем размер
            rs = _resize(roi, max(1, int(w / f)), max(1, int(h / f)))
            rs = cv2.GaussianBlur(rs, (0, 0), sig / f)
            roi = _resize(rs, w, h, cv2.INTER_LINEAR)
        else:
            roi = cv2.GaussianBlur(roi, (0, 0), sig)
        mf *= roi

    mf = cv2.GaussianBlur(mf, (0, 0), max(1.0, short * 0.003))
    return np.clip(mf, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Ретушь кожи
# ----------------------------------------------------------------------------
def _resize(img: np.ndarray, w: int, h: int, interp=cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def smooth_skin(img: np.ndarray, mask: np.ndarray, smooth: float, texture: float) -> np.ndarray:
    """Частотное разложение: сглаживаем низкие частоты, текстуру сохраняем по слайдеру."""
    h, w = img.shape[:2]
    long_ = max(h, w)
    s = float(np.clip(smooth / 100.0, 0.0, 1.0))
    tex = float(np.clip(texture / 100.0, 0.0, 1.0))

    sigma_d = max(1.0, 0.001 * long_)
    base = cv2.GaussianBlur(img, (0, 0), sigma_d)
    detail = img - base

    scale = min(1.0, 1000.0 / long_)
    small = _resize(base, max(1, int(w * scale)), max(1, int(h * scale))) if scale < 1 else base

    sigma_c = 0.03 + 0.14 * s
    sigma_s = 3.0 + 5.0 * s
    passes = 1 + int(round(s * 2.5))
    low = small
    for _ in range(passes):
        low = cv2.bilateralFilter(low, 9, sigma_c, sigma_s)
    if scale < 1:
        low = _resize(low, w, h, cv2.INTER_LINEAR)

    smoothed = low + detail * tex
    alpha = (mask * (0.25 + 0.75 * s))[..., None]
    return img * (1.0 - alpha) + smoothed * alpha


def remove_blemishes(img: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Ищет мелкие тёмные/красные пятна на коже и заменяет их соседним тоном."""
    h, w = img.shape[:2]
    long_ = max(h, w)
    scale = min(1.0, 1500.0 / long_)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = _resize(img, sw, sh) if scale < 1 else img
    small8 = np.clip(small * 255.0 + 0.5, 0, 255).astype(np.uint8)
    smask = _resize(mask, sw, sh) if scale < 1 else mask

    short = min(sw, sh)
    k = _odd(short * 0.016, 7)
    lab = cv2.cvtColor(small8, cv2.COLOR_BGR2LAB)
    L, A = lab[..., 0], lab[..., 1]
    med_l = cv2.medianBlur(L, k)
    med_a = cv2.medianBlur(A, k)

    sens = float(np.clip(amount / 100.0, 0.0, 1.0))
    thr_l = 30.0 - 21.0 * sens
    thr_a = 14.0 - 8.0 * sens
    dl = med_l.astype(np.int16) - L.astype(np.int16)
    da = A.astype(np.int16) - med_a.astype(np.int16)
    spots = (((dl > thr_l) | (da > thr_a)) & (smask > 0.85)).astype(np.uint8)

    # оставляем только компактные небольшие пятна (не брови, не губы, не глаза)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
    max_area = (k * 0.5) ** 2 * 3.14
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 2 or area > max_area:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 3.0:
            continue
        if area / float(bw * bh) < 0.35:
            continue
        keep[i] = 1
    spots = keep[labels].astype(np.uint8)
    if not spots.any():
        return img

    grow = _odd(short * 0.004, 3)
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    soft = cv2.GaussianBlur(spots.astype(np.float32), (0, 0), max(1.0, grow * 0.5))
    soft = np.clip(soft * 1.5, 0, 1)

    repl = cv2.medianBlur(small8, _odd(k * 0.9, 5)).astype(np.float32) / 255.0
    if scale < 1:
        soft = _resize(soft, w, h, cv2.INTER_LINEAR)
        repl = _resize(repl, w, h, cv2.INTER_LINEAR)
    soft = soft[..., None]
    return img * (1.0 - soft) + repl * soft


def even_skin_tone(img: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Выравнивает цветность кожи (покраснения, пятна) — нормализованная свёртка по маске."""
    h, w = img.shape[:2]
    scale = min(1.0, 800.0 / max(h, w))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = _resize(img, sw, sh) if scale < 1 else img
    smask = _resize(mask, sw, sh) if scale < 1 else mask

    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_BGR2LAB)
    sig = max(2.0, min(sw, sh) * 0.025)
    den = cv2.GaussianBlur(smask, (0, 0), sig) + 1e-4
    out = lab.copy()
    for c in (1, 2):
        avg = cv2.GaussianBlur(lab[..., c] * smask, (0, 0), sig) / den
        out[..., c] = lab[..., c] + (avg - lab[..., c])
    lab_new = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    lab_old = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    delta = lab_new - lab_old
    if scale < 1:
        delta = _resize(delta, w, h, cv2.INTER_LINEAR)
    t = float(np.clip(amount / 100.0, 0.0, 1.0)) * 0.85
    return img + delta * (mask * t)[..., None]


# ----------------------------------------------------------------------------
# Цвет и свет
# ----------------------------------------------------------------------------
def _tone_and_color(img: np.ndarray, p: Params) -> np.ndarray:
    if p.exposure:
        img = img * (2.0 ** p.exposure)
    if p.temperature or p.tint:
        t, ti = p.temperature / 100.0, p.tint / 100.0
        gains = np.array([1.0 - 0.22 * t, 1.0 - 0.14 * ti, 1.0 + 0.22 * t], np.float32)  # B,G,R
        img = img * gains
    img = np.clip(img, 0.0, 1.0)

    if p.shadows or p.highlights:
        lum = _luma(img)
        sh_w = (1.0 - lum) ** 2
        hi_w = lum ** 2
        gain = 1.0 + (p.shadows / 100.0) * 0.8 * sh_w + (p.highlights / 100.0) * 0.5 * hi_w
        img = np.clip(img * gain[..., None], 0.0, 1.0)

    if p.contrast:
        c = p.contrast / 100.0
        img = np.clip((img - 0.5) * (1.0 + 0.8 * c) + 0.5, 0.0, 1.0)
    return img


def _saturation(img: np.ndarray, p: Params) -> np.ndarray:
    if not (p.saturation or p.vibrance):
        return img
    gray = _luma(img)[..., None]
    f = 1.0 + p.saturation / 100.0
    if p.vibrance:
        cur = (img.max(axis=2) - img.min(axis=2))[..., None]
        f = f * (1.0 + (p.vibrance / 100.0) * (1.0 - cur))
    return np.clip(gray + (img - gray) * f, 0.0, 1.0)


def _clarity(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    sigma = max(2.0, 0.012 * max(h, w))
    f = max(1.0, sigma / 6.0)  # большой радиус считаем на уменьшенной копии — иначе очень медленно
    if f > 1.0:
        small = _resize(img, max(1, int(w / f)), max(1, int(h / f)))
        blur = cv2.GaussianBlur(small, (0, 0), sigma / f)
        blur = _resize(blur, w, h, cv2.INTER_LINEAR)
    else:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
    lum = _luma(img)
    mid = (1.0 - np.abs(lum * 2.0 - 1.0) ** 2)[..., None]  # сильнее в полутонах
    return np.clip(img + (img - blur) * (amount / 100.0) * 0.8 * mid, 0.0, 1.0)


def _sharpen(img: np.ndarray, amount: float, mask: Optional[np.ndarray]) -> np.ndarray:
    h, w = img.shape[:2]
    sigma = max(0.8, 0.0007 * max(h, w))
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    k = (amount / 100.0) * 1.6
    if mask is not None:
        k = k * (1.0 - 0.6 * mask)[..., None]  # кожу заостряем слабее
    return np.clip(img + (img - blur) * k, 0.0, 1.0)


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt(((xx - n / 2) / (n / 2)) ** 2 + ((yy - n / 2) / (n / 2)) ** 2) / np.sqrt(2.0)
    t = np.clip((r - 0.35) / 0.65, 0.0, 1.0)
    fall = t * t * (3 - 2 * t)
    mult = 1.0 - (amount / 100.0) * 0.75 * fall
    mult = cv2.resize(mult, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(img * mult[..., None], 0.0, 1.0)


# ----------------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------------
def process(img_u8: np.ndarray, p: Params,
            faces: Optional[Sequence[FaceBox]] = None) -> np.ndarray:
    """BGR uint8 -> BGR uint8. faces — нормализованные боксы (можно передать из кэша)."""
    src = img_u8
    img = src.astype(np.float32) / 255.0

    mask = None
    if p.needs_skin_mask() or p.sharpen > 0:
        if faces is None:
            faces = detect_faces(src)
        if p.needs_skin_mask():
            mask = skin_mask(src, faces, p.faces_only)

    img = _tone_and_color(img, p)

    if mask is not None:
        if p.blemish > 0:
            img = remove_blemishes(img, mask, p.blemish)
        if p.skin_smooth > 0:
            img = smooth_skin(img, mask, p.skin_smooth, p.skin_texture)
        if p.even_tone > 0:
            img = even_skin_tone(img, mask, p.even_tone)

    img = _saturation(img, p)
    if p.clarity:
        img = _clarity(img, p.clarity)
    if p.sharpen > 0:
        img = _sharpen(img, p.sharpen, mask)
    if p.vignette:
        img = _vignette(img, p.vignette)

    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
