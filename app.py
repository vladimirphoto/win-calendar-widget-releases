"""Lumina Retouch — настольное приложение для ретуши фото (PySide6 + OpenCV).

Запуск:  python app.py [файлы или папки ...]
"""
from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import (QEvent, QObject, QPointF, QRectF, QRunnable, QSize, Qt,
                            QThread, QThreadPool, QTimer, Signal)
from PySide6.QtGui import (QAction, QColor, QFont, QIcon, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressDialog, QPushButton, QScrollArea, QSlider,
                               QSpinBox, QStatusBar, QToolBar, QVBoxLayout, QWidget)

import engine
import io_utils
from engine import Params, PRESETS

APP_NAME = "Lumina Retouch"
PREVIEW_LONG_SIDE = 2000
PREVIEW_CACHE = 12
USER_DIR = Path.home() / ".lumina_retouch"
USER_PRESETS = USER_DIR / "presets.json"

# (поле Params, подпись, минимум, максимум, масштаб значения, по умолчанию-подсказка)
SECTIONS = [
    ("Кожа", [
        ("skin_smooth", "Сглаживание", 0, 100, 1),
        ("skin_texture", "Сохранить текстуру", 0, 100, 1),
        ("blemish", "Удаление дефектов", 0, 100, 1),
        ("even_tone", "Выравнивание тона", 0, 100, 1),
    ]),
    ("Свет", [
        ("exposure", "Экспозиция (EV)", -200, 200, 100),
        ("contrast", "Контраст", -100, 100, 1),
        ("highlights", "Света", -100, 100, 1),
        ("shadows", "Тени", -100, 100, 1),
    ]),
    ("Цвет", [
        ("temperature", "Температура", -100, 100, 1),
        ("tint", "Оттенок", -100, 100, 1),
        ("saturation", "Насыщенность", -100, 100, 1),
        ("vibrance", "Красочность", -100, 100, 1),
    ]),
    ("Детали и эффекты", [
        ("clarity", "Чёткость (clarity)", -100, 100, 1),
        ("sharpen", "Резкость", 0, 100, 1),
        ("vignette", "Виньетка", -100, 100, 1),
    ]),
]

QSS = """
QMainWindow, QDialog { background: #1b1c20; }
QWidget { color: #d9dbe0; font-size: 13px; }
QToolBar { background: #232429; border: none; border-bottom: 1px solid #2f3037; spacing: 6px; padding: 5px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 9px; }
QToolButton:hover { background: #2e3038; }
QToolButton:checked { background: #3a4260; border-color: #5a6bb5; }
QListWidget { background: #17181b; border: none; outline: 0; }
QListWidget::item { padding: 4px; border-radius: 6px; }
QListWidget::item:selected { background: #2f3a63; }
QScrollArea { border: none; background: #1f2024; }
QGroupBox, QFrame#section { background: #25262c; border-radius: 8px; }
QLabel#sectionTitle { font-weight: 600; color: #aab3e8; padding-top: 6px; }
QSlider::groove:horizontal { height: 4px; background: #3a3c46; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #6f86ff; border-radius: 2px; }
QSlider::handle:horizontal { background: #e8ebff; width: 14px; margin: -6px 0; border-radius: 7px; }
QPushButton { background: #333644; border: 1px solid #44485a; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #3d4152; }
QPushButton#primary { background: #5468ff; border-color: #5468ff; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #6577ff; }
QComboBox, QLineEdit, QSpinBox { background: #2a2c35; border: 1px solid #3f4252; border-radius: 6px; padding: 4px 8px; }
QStatusBar { background: #232429; color: #9aa0b4; }
QProgressDialog { background: #1b1c20; }
"""


# ----------------------------------------------------------------------------
# Фоновые задачи
# ----------------------------------------------------------------------------
class Bus(QObject):
    """Живёт в главном потоке; через него воркеры возвращают результаты."""
    render_done = Signal(object)
    load_done = Signal(object)
    thumb_done = Signal(object)
    error = Signal(str)


class Task(QRunnable):
    def __init__(self, fn: Callable[[], object], ok_signal, err_signal):
        super().__init__()
        self.fn, self.ok, self.err = fn, ok_signal, err_signal

    def run(self) -> None:
        try:
            self.ok.emit(self.fn())
        except Exception:
            self.err.emit(traceback.format_exc())


def bgr_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_BGR888).copy()


class Photo:
    def __init__(self, path: Path):
        self.path = path
        self.params = Params()
        self.src: Optional[np.ndarray] = None       # превью-оригинал (BGR)
        self.size: Optional[tuple] = None            # (w, h) оригинала
        self.meta: Optional[io_utils.Meta] = None
        self.faces = None                            # None = ещё не искали
        self.item: Optional[QListWidgetItem] = None


# ----------------------------------------------------------------------------
# Холст с «до / после»
# ----------------------------------------------------------------------------
class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.before: Optional[QImage] = None
        self.after: Optional[QImage] = None
        self.split_mode = False
        self.hold_before = False
        self.split = 0.5
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self._drag: Optional[str] = None
        self._last = QPointF()
        self.message = "Перетащите фото сюда или нажмите «Открыть»"

    # --- данные ---
    def set_images(self, before: Optional[QImage], after: Optional[QImage], keep_view=True):
        reset = (not keep_view) or self.before is None or (
            before is not None and self.before is not None and before.size() != self.before.size())
        self.before, self.after = before, after
        if reset:
            self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    def set_after(self, after: QImage):
        self.after = after
        self.update()

    # --- геометрия ---
    def _rect(self) -> Optional[QRectF]:
        img = self.after or self.before
        if img is None:
            return None
        iw, ih = img.width(), img.height()
        fit = min(self.width() / iw, self.height() / ih) * 0.97
        s = fit * self.zoom
        cx, cy = self.width() / 2 + self.offset.x(), self.height() / 2 + self.offset.y()
        return QRectF(cx - iw * s / 2, cy - ih * s / 2, iw * s, ih * s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#121316"))
        r = self._rect()
        if r is None:
            p.setPen(QColor("#6d7287"))
            p.setFont(QFont(self.font().family(), 13))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self.hold_before and self.before is not None:
            p.drawImage(r, self.before)
            self._tag(p, r, "ДО", right=False)
            return
        if self.after is not None:
            p.drawImage(r, self.after)
        if self.split_mode and self.before is not None:
            x = r.left() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), x - r.left(), r.height()))
            p.drawImage(r, self.before)
            p.restore()
            p.setPen(QPen(QColor("white"), 2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setBrush(QColor("white"))
            p.setPen(Qt.NoPen)
            cy = r.center().y()
            p.drawEllipse(QPointF(x, cy), 11, 11)
            p.setPen(QColor("#222"))
            p.drawText(QRectF(x - 11, cy - 11, 22, 22), Qt.AlignCenter, "↔")
            self._tag(p, r, "ДО", right=False)
            self._tag(p, r, "ПОСЛЕ", right=True)

    @staticmethod
    def _tag(p: QPainter, r: QRectF, text: str, right: bool):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        w = 12 + 8 * len(text)
        x = r.right() - w - 10 if right else r.left() + 10
        box = QRectF(x, r.top() + 10, w, 24)
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QColor("white"))
        p.drawText(box, Qt.AlignCenter, text)

    # --- мышь ---
    def wheelEvent(self, e):
        r = self._rect()
        if r is None:
            return
        old = self.zoom
        self.zoom = float(np.clip(self.zoom * (1.2 if e.angleDelta().y() > 0 else 1 / 1.2), 1.0, 16.0))
        ratio = self.zoom / old
        c = QPointF(self.width() / 2, self.height() / 2)
        pos = e.position()
        self.offset = pos - c - (pos - c - self.offset) * ratio
        if self.zoom == 1.0:
            self.offset = QPointF(0, 0)
        self.update()

    def mousePressEvent(self, e):
        r = self._rect()
        if r is None or e.button() != Qt.LeftButton:
            return
        self._last = e.position()
        if self.split_mode and abs(e.position().x() - (r.left() + r.width() * self.split)) < 14:
            self._drag = "split"
        else:
            self._drag = "pan"
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        r = self._rect()
        if self._drag == "split" and r is not None:
            self.split = float(np.clip((e.position().x() - r.left()) / r.width(), 0.0, 1.0))
            self.update()
        elif self._drag == "pan":
            if self.zoom > 1.0:
                self.offset = self.offset + (e.position() - self._last)
                self.update()
            self._last = e.position()
        elif r is not None and self.split_mode:
            near = abs(e.position().x() - (r.left() + r.width() * self.split)) < 14
            self.setCursor(Qt.SplitHCursor if near else Qt.ArrowCursor)

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, _):
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()


# ----------------------------------------------------------------------------
# Панель ползунков
# ----------------------------------------------------------------------------
class SliderRow(QWidget):
    changed = Signal(str, float)

    def __init__(self, name: str, label: str, lo: int, hi: int, scale: int, default: float):
        super().__init__()
        self.name, self.scale, self.default = name, scale, default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(0)
        head = QHBoxLayout()
        self.title = QLabel(label)
        self.value_label = QLabel("0")
        self.value_label.setAlignment(Qt.AlignRight)
        self.value_label.setStyleSheet("color:#9aa6ff;")
        head.addWidget(self.title)
        head.addWidget(self.value_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        lay.addLayout(head)
        lay.addWidget(self.slider)
        self.slider.valueChanged.connect(self._on_change)
        self.title.setToolTip("Двойной щелчок по ползунку — сброс")
        self.slider.mouseDoubleClickEvent = lambda ev: self.reset()  # type: ignore[assignment]

    def _fmt(self, v: float) -> str:
        return f"{v:.2f}" if self.scale != 1 else f"{int(round(v))}"

    def _on_change(self, raw: int):
        v = raw / self.scale
        self.value_label.setText(self._fmt(v))
        self.changed.emit(self.name, v)

    def set_value(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.scale)))
        self.slider.blockSignals(False)
        self.value_label.setText(self._fmt(v))

    def reset(self):
        self.slider.setValue(int(round(self.default * self.scale)))


class AdjustPanel(QScrollArea):
    param_changed = Signal(str, float)
    flag_changed = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(290)
        self.setMaximumWidth(340)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(12, 8, 12, 12)
        self.rows: Dict[str, SliderRow] = {}
        defaults = Params()
        for title, items in SECTIONS:
            t = QLabel(title.upper())
            t.setObjectName("sectionTitle")
            lay.addWidget(t)
            for (name, label, lo, hi, scale) in items:
                row = SliderRow(name, label, lo, hi, scale, getattr(defaults, name))
                row.changed.connect(self.param_changed)
                self.rows[name] = row
                lay.addWidget(row)
            if title == "Кожа":
                self.faces_only = QCheckBox("Только люди в кадре (если найдены лица)")
                self.faces_only.setChecked(True)
                self.faces_only.toggled.connect(lambda v: self.flag_changed.emit("faces_only", v))
                lay.addWidget(self.faces_only)
        lay.addStretch(1)
        self.setWidget(body)

    def set_params(self, p: Params):
        for name, row in self.rows.items():
            row.set_value(getattr(p, name))
        self.faces_only.blockSignals(True)
        self.faces_only.setChecked(p.faces_only)
        self.faces_only.blockSignals(False)


# ----------------------------------------------------------------------------
# Экспорт
# ----------------------------------------------------------------------------
class ExportDialog(QDialog):
    def __init__(self, parent, n_all: int, default_dir: Path):
        super().__init__(parent)
        self.setWindowTitle("Экспорт")
        self.setMinimumWidth(460)
        form = QFormLayout(self)
        self.scope = QComboBox()
        self.scope.addItems(["Выбранное фото", f"Все фото ({n_all})"])
        self.fmt = QComboBox()
        self.fmt.addItems(["JPEG", "PNG", "TIFF", "WEBP"])
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        self.long_side = QSpinBox()
        self.long_side.setRange(0, 20000)
        self.long_side.setSpecialValueText("оригинал")
        self.long_side.setSuffix(" px")
        self.suffix = QLineEdit("_retouch")
        self.folder = QLineEdit(str(default_dir))
        browse = QPushButton("…")
        browse.setFixedWidth(34)
        browse.clicked.connect(self._browse)
        frow = QHBoxLayout()
        frow.addWidget(self.folder)
        frow.addWidget(browse)
        form.addRow("Что экспортировать:", self.scope)
        form.addRow("Формат:", self.fmt)
        form.addRow("Качество (JPEG/WEBP):", self.quality)
        form.addRow("Длинная сторона:", self.long_side)
        form.addRow("Суффикс имени:", self.suffix)
        form.addRow("Папка:", frow)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Экспортировать")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для экспорта", self.folder.text())
        if d:
            self.folder.setText(d)

    def values(self) -> dict:
        ext = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}[self.fmt.currentText()]
        return dict(all=self.scope.currentIndex() == 1, ext=ext, quality=self.quality.value(),
                    long_side=self.long_side.value(), suffix=self.suffix.text(),
                    folder=Path(self.folder.text()))


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    finished_all = Signal(list)

    def __init__(self, photos: List[Photo], opts: dict):
        super().__init__()
        self.photos, self.opts = photos, opts
        self.cancel = False

    def _one(self, ph: Photo) -> Optional[str]:
        if self.cancel:
            return None
        o = self.opts
        img, meta = io_utils.load_image(ph.path)
        out = engine.process(img, ph.params, ph.faces)
        if o["long_side"]:
            out = io_utils.downscale_to(out, o["long_side"])
        target = o["folder"] / f"{ph.path.stem}{o['suffix']}{o['ext']}"
        i = 1
        while target.exists():
            target = o["folder"] / f"{ph.path.stem}{o['suffix']}_{i}{o['ext']}"
            i += 1
        io_utils.save_image(target, out, meta, o["quality"])
        return None

    def run(self):
        errors: List[str] = []
        done = 0
        total = len(self.photos)
        self.opts["folder"].mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(self._one, ph): ph for ph in self.photos}
            for f in as_completed(futs):
                ph = futs[f]
                try:
                    f.result()
                except Exception as exc:
                    errors.append(f"{ph.path.name}: {exc}")
                done += 1
                self.progress.emit(done, total, ph.path.name)
        self.finished_all.emit(errors)


# ----------------------------------------------------------------------------
# Главное окно
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.photos: List[Photo] = []
        self.current: Optional[Photo] = None
        self.clipboard: Optional[Params] = None
        self.cache_order: List[Photo] = []
        self.user_presets: Dict[str, dict] = self._load_user_presets()
        self.export_worker: Optional[ExportWorker] = None

        self.bus = Bus()
        self.bus.render_done.connect(self._on_render_done)
        self.bus.load_done.connect(self._on_load_done)
        self.bus.thumb_done.connect(self._on_thumb_done)
        self.bus.error.connect(self._on_error)
        self.render_pool = QThreadPool()
        self.render_pool.setMaxThreadCount(1)
        self.io_pool = QThreadPool()
        self.io_pool.setMaxThreadCount(2)
        self._render_busy = False
        self._render_dirty = False
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(30)
        self.debounce.timeout.connect(self._start_render)

        self._build_ui()
        self.statusBar().showMessage("Готово. Откройте фото или перетащите их в окно.")

    # ---- UI ----
    def _build_ui(self):
        self.canvas = Canvas()
        self.panel = AdjustPanel()
        self.panel.param_changed.connect(self._on_param)
        self.panel.flag_changed.connect(self._on_flag)

        self.list = QListWidget()
        self.list.setIconSize(QSize(72, 72))
        self.list.setFixedWidth(150)
        self.list.currentItemChanged.connect(self._on_select)

        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.list)
        row.addWidget(self.canvas, 1)
        row.addWidget(self.panel)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        tb = QToolBar("Главная")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        def act(text: str, slot, shortcut: str | None = None, checkable=False, tip: str = ""):
            a = QAction(text, self)
            a.setCheckable(checkable)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("Открыть файлы", self.open_files, "Ctrl+O")
        act("Открыть папку", self.open_folder, "Ctrl+Shift+O")
        tb.addSeparator()
        act("Экспорт…", self.export_dialog, "Ctrl+E")
        tb.addSeparator()
        act("Копировать настройки", self.copy_params, "Ctrl+Shift+C")
        act("Вставить", self.paste_params, "Ctrl+Shift+V")
        act("Применить ко всем", self.apply_to_all)
        act("Сбросить", self.reset_current, "Ctrl+R")
        tb.addSeparator()
        self.sync_box = QCheckBox("Синхронизировать со всеми")
        self.sync_box.setToolTip("Любое движение ползунка сразу применяется ко всем фото в списке")
        tb.addWidget(self.sync_box)
        tb.addSeparator()
        self.split_act = act("До / После", self._toggle_split, "Ctrl+B", checkable=True,
                             tip="Разделённый просмотр. Пробел (удерживать) — показать оригинал")
        tb.addSeparator()
        tb.addWidget(QLabel(" Пресет: "))
        self.preset_box = QComboBox()
        self.preset_box.setMinimumWidth(160)
        self._refresh_presets()
        self.preset_box.activated.connect(self._apply_preset)
        tb.addWidget(self.preset_box)
        act("Сохранить пресет", self.save_preset)

        QApplication.instance().installEventFilter(self)

    # ---- события клавиатуры (пробел = показать оригинал) ----
    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_Space \
                and not ev.isAutoRepeat() and self.isActiveWindow():
            self.canvas.hold_before = ev.type() == QEvent.KeyPress
            self.canvas.update()
            return True
        return super().eventFilter(obj, ev)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([Path(u.toLocalFile()) for u in e.mimeData().urls()])

    # ---- файлы ----
    def open_files(self):
        exts = " ".join(f"*{x}" for x in sorted(io_utils.SUPPORTED_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, "Открыть фото", "", f"Изображения ({exts})")
        if files:
            self.add_paths([Path(f) for f in files])

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с фото")
        if d:
            self.add_paths([Path(d)])

    def add_paths(self, paths: List[Path]):
        found: List[Path] = []
        for p in paths:
            if p.is_dir():
                found += sorted(q for q in p.iterdir() if q.is_file() and io_utils.is_supported(q))
            elif p.is_file() and io_utils.is_supported(p):
                found.append(p)
        known = {ph.path for ph in self.photos}
        first_new = None
        for p in found:
            if p in known:
                continue
            ph = Photo(p)
            item = QListWidgetItem(p.name)
            item.setSizeHint(QSize(130, 100))
            pm = QPixmap(72, 72)
            pm.fill(QColor("#2a2c35"))
            item.setIcon(QIcon(pm))
            item.setData(Qt.UserRole, len(self.photos))
            ph.item = item
            self.photos.append(ph)
            self.list.addItem(item)
            first_new = first_new or item
            self._queue_thumb(ph)
        if first_new is not None and self.current is None:
            self.list.setCurrentItem(first_new)
        self.statusBar().showMessage(f"Фото в списке: {len(self.photos)}")

    def _queue_thumb(self, ph: Photo):
        def fn():
            return ph, io_utils.load_thumbnail(ph.path, 144)
        self.io_pool.start(Task(fn, self.bus.thumb_done, self.bus.error))

    def _on_thumb_done(self, res):
        ph, arr = res
        pm = QPixmap.fromImage(bgr_to_qimage(arr)).scaled(72, 72, Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation)
        if ph.item is not None:
            ph.item.setIcon(QIcon(pm))

    # ---- выбор фото ----
    def _on_select(self, item: Optional[QListWidgetItem], _prev=None):
        if item is None:
            return
        ph = self.photos[item.data(Qt.UserRole)]
        self.current = ph
        self.panel.set_params(ph.params)
        if ph.src is not None:
            self._show_current()
            return
        self.canvas.message = "Загрузка…"
        self.canvas.set_images(None, None)
        self.statusBar().showMessage(f"Загружаю {ph.path.name}…")

        def fn():
            img, meta = io_utils.load_image(ph.path)
            h, w = img.shape[:2]
            return ph, io_utils.downscale_to(img, PREVIEW_LONG_SIDE), meta, (w, h)
        self.io_pool.start(Task(fn, self.bus.load_done, self.bus.error))

    def _on_load_done(self, res):
        ph, src, meta, size = res
        ph.src, ph.meta, ph.size = src, meta, size
        self.cache_order = [p for p in self.cache_order if p is not ph] + [ph]
        while len(self.cache_order) > PREVIEW_CACHE:
            old = self.cache_order.pop(0)
            if old is not self.current:
                old.src = None
        if ph is self.current:
            self._show_current()

    def _show_current(self):
        ph = self.current
        if ph is None or ph.src is None:
            return
        before = bgr_to_qimage(ph.src)
        self.canvas.set_images(before, before, keep_view=False)
        self.statusBar().showMessage(
            f"{ph.path.name}   {ph.size[0]}×{ph.size[1]}   (превью {ph.src.shape[1]}×{ph.src.shape[0]})")
        self.request_render()

    # ---- рендер превью ----
    def request_render(self):
        self.debounce.start()

    def _start_render(self):
        ph = self.current
        if ph is None or ph.src is None:
            return
        if self._render_busy:
            self._render_dirty = True
            return
        self._render_busy = True
        self._render_dirty = False
        params = ph.params.copy()

        def fn():
            faces = ph.faces
            if faces is None and (params.needs_skin_mask() or params.sharpen > 0):
                faces = engine.detect_faces(ph.src)
            out = engine.process(ph.src, params, faces)
            return ph, faces, out

        self.render_pool.start(Task(fn, self.bus.render_done, self.bus.error))

    def _on_render_done(self, res):
        ph, faces, out = res
        self._render_busy = False
        if faces is not None:
            ph.faces = faces
        if ph is self.current:
            self.canvas.set_after(bgr_to_qimage(out))
        if self._render_dirty:
            self._start_render()

    def _on_error(self, text: str):
        self._render_busy = False
        self.statusBar().showMessage("Ошибка — подробности в консоли")
        print(text, file=sys.stderr)

    # ---- параметры ----
    def _mark_edited(self, ph: Photo):
        if ph.item is not None:
            edited = ph.params != Params()
            ph.item.setText(("● " if edited else "") + ph.path.name)

    def _on_param(self, name: str, value: float):
        if self.current is None:
            return
        targets = self.photos if self.sync_box.isChecked() else [self.current]
        for ph in targets:
            setattr(ph.params, name, value)
            self._mark_edited(ph)
        self.request_render()

    def _on_flag(self, name: str, value: bool):
        if self.current is None:
            return
        targets = self.photos if self.sync_box.isChecked() else [self.current]
        for ph in targets:
            setattr(ph.params, name, value)
        self.request_render()

    def copy_params(self):
        if self.current:
            self.clipboard = self.current.params.copy()
            self.statusBar().showMessage("Настройки скопированы")

    def paste_params(self):
        if self.current and self.clipboard:
            self.current.params = self.clipboard.copy()
            self.panel.set_params(self.current.params)
            self._mark_edited(self.current)
            self.request_render()

    def apply_to_all(self):
        if not self.current:
            return
        for ph in self.photos:
            ph.params = self.current.params.copy()
            self._mark_edited(ph)
        self.statusBar().showMessage(f"Настройки применены к {len(self.photos)} фото")

    def reset_current(self):
        if self.current:
            self.current.params = Params()
            self.panel.set_params(self.current.params)
            self._mark_edited(self.current)
            self.request_render()

    def _toggle_split(self):
        self.canvas.split_mode = self.split_act.isChecked()
        self.canvas.update()

    # ---- пресеты ----
    def _load_user_presets(self) -> Dict[str, dict]:
        try:
            return json.loads(USER_PRESETS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _refresh_presets(self):
        self.preset_box.clear()
        self.preset_box.addItem("— выбрать —")
        for n in PRESETS:
            self.preset_box.addItem(n)
        for n in self.user_presets:
            self.preset_box.addItem("★ " + n)

    def _apply_preset(self, idx: int):
        if idx <= 0 or self.current is None:
            return
        name = self.preset_box.itemText(idx)
        if name.startswith("★ "):
            p = Params.from_dict(self.user_presets[name[2:]])
        else:
            p = PRESETS[name].copy()
        targets = self.photos if self.sync_box.isChecked() else [self.current]
        for ph in targets:
            ph.params = p.copy()
            self._mark_edited(ph)
        self.panel.set_params(self.current.params)
        self.request_render()

    def save_preset(self):
        if self.current is None:
            return
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:")
        name = name.strip()
        if not ok or not name:
            return
        self.user_presets[name] = self.current.params.to_dict()
        USER_DIR.mkdir(parents=True, exist_ok=True)
        USER_PRESETS.write_text(json.dumps(self.user_presets, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        self._refresh_presets()
        self.statusBar().showMessage(f"Пресет «{name}» сохранён")

    # ---- экспорт ----
    def export_dialog(self):
        if not self.photos:
            QMessageBox.information(self, APP_NAME, "Сначала откройте фото.")
            return
        base = (self.current or self.photos[0]).path.parent / "Lumina Export"
        dlg = ExportDialog(self, len(self.photos), base)
        if dlg.exec() != QDialog.Accepted:
            return
        o = dlg.values()
        items = self.photos if o["all"] else ([self.current] if self.current else [])
        if not items:
            return
        prog = QProgressDialog("Экспорт…", "Отмена", 0, len(items), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        worker = ExportWorker(list(items), o)
        self.export_worker = worker

        def on_progress(done: int, total: int, name: str):
            prog.setLabelText(f"{name}  ({done}/{total})")
            prog.setValue(done)

        def on_finished(errors: list):
            prog.close()
            if errors:
                QMessageBox.warning(self, APP_NAME, "Часть файлов не удалось сохранить:\n\n"
                                    + "\n".join(errors[:15]))
            else:
                self.statusBar().showMessage(f"Готово: {len(items)} фото → {o['folder']}")
                QMessageBox.information(self, APP_NAME, f"Экспорт завершён.\n\n{o['folder']}")

        worker.progress.connect(on_progress)
        worker.finished_all.connect(on_finished)
        prog.canceled.connect(lambda: setattr(worker, "cancel", True))
        worker.start()


def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#1b1c20"))
    pal.setColor(QPalette.WindowText, QColor("#d9dbe0"))
    pal.setColor(QPalette.Base, QColor("#17181b"))
    pal.setColor(QPalette.AlternateBase, QColor("#1f2024"))
    pal.setColor(QPalette.Text, QColor("#d9dbe0"))
    pal.setColor(QPalette.Button, QColor("#2a2c35"))
    pal.setColor(QPalette.ButtonText, QColor("#d9dbe0"))
    pal.setColor(QPalette.Highlight, QColor("#5468ff"))
    pal.setColor(QPalette.HighlightedText, QColor("white"))
    pal.setColor(QPalette.ToolTipBase, QColor("#2a2c35"))
    pal.setColor(QPalette.ToolTipText, QColor("#d9dbe0"))
    app.setPalette(pal)
    app.setStyleSheet(QSS)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_dark_palette(app)
    win = MainWindow()
    win.show()
    args = [Path(a) for a in sys.argv[1:]]
    if args:
        win.add_paths(args)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
