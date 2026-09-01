from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from model.background_worker import CallableWorker
from model.voice_input import (
    VOICE_AVAILABLE,
    MicrophoneWorker,
    TranscriptionWorker,
    list_input_devices,
    select_input_device,
    selected_input_device,
)
from utils.app_config import (
    OLLAMA_MODEL,
    discover_local_gguf_files,
    gguf_model_tag,
    import_gguf_model,
    list_ollama_models,
)

_USER_BUBBLE_WIDTH_FRACTION = 0.7  # user bubbles can grow up to 70% of the panel width

# Result strings that just restate what the tool call's own summary line
# already says (e.g. "Apply viability preset" -> "Applied 'viability'
# preset.") add nothing when shown underneath it, so only surface a result
# when it looks like something went wrong.
_FAILURE_MARKERS = ("failed", "not found", "not loaded", "out of range", "unknown tool")

# Friendly one-line phrasing for tool calls, so the panel reads like "Load
# subject 3" instead of raw JSON — the exact parsed arguments are still
# available on click, mirroring how agentic coding UIs collapse tool calls
# to a summary line with an expandable detail underneath.
_TOOL_SUMMARY_FORMATTERS = {
    "load_subject": lambda a: f"Load subject {a.get('subject_number')}",
    "run_segmentation": lambda a: f"Run {a.get('modality')} segmentation",
    "apply_preset": lambda a: f"Apply {a.get('preset')} preset",
    "display_series": lambda a: (
        f"Show {a.get('series_key')} in panel {a.get('panel_idx')}"
        # "transverse" is the default axis, not worth echoing back.
        + (f" ({a['axis']})" if a.get("axis") not in (None, "", "transverse") else "")
    ),
    "set_time_frame": lambda a: f"Jump to time frame {a.get('time_index')}",
    "save_segmentation": lambda a: "Save segmentation",
}


def _tool_call_summary(name: str, arguments: dict) -> str:
    formatter = _TOOL_SUMMARY_FORMATTERS.get(name)
    if formatter is None:
        return name.replace("_", " ").capitalize()
    try:
        return formatter(arguments)
    except Exception:
        return name.replace("_", " ").capitalize()


class _Bubble(QFrame):
    """A single chat message: user (right-aligned, accent, capped at a
    fraction of the panel width), assistant/system (fill the panel width)."""

    _MARGINS = 20  # left+right content margins/padding, roughly

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.setProperty("role", f"chatBubble{role.capitalize()}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(0)
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._label)

    def set_text(self, text: str):
        self._label.setText(text)

    def natural_width(self) -> int:
        """Width needed to render the current text on one line — QLabel's own
        sizeHint() is unreliable for this once word-wrap is on, so measure
        directly with font metrics instead."""
        return self._label.fontMetrics().horizontalAdvance(self._label.text()) + self._MARGINS

    def apply_width_cap(self, cap: int):
        """Fix the bubble's width to whatever is smaller: its natural
        (unwrapped) width, or `cap`. Short messages hug their content; long
        ones wrap and grow up to the cap — never wider."""
        width = min(cap, max(self.natural_width(), 40))
        self.setFixedWidth(width)
        # Also fix the label's own width — otherwise its wrapped sizeHint
        # can lag a layout pass behind the bubble's.
        self._label.setFixedWidth(max(width - self._MARGINS, 10))


class _ToolCallBubble(QFrame):
    """Collapsed-by-default tool call: a clickable summary line ('Load
    subject 3'), expanding on click to show the exact arguments the model
    passed and the result the tool returned."""

    def __init__(self, name: str, arguments: dict, parent=None):
        super().__init__(parent)
        self.setProperty("role", "chatBubbleTool")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)

        self._header = QPushButton(_tool_call_summary(name, arguments))
        self._header.setProperty("role", "toolCallHeader")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        self._detail = QLabel()
        self._detail.setProperty("role", "toolCallDetail")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._detail.setText(f"{name}({json.dumps(arguments)})")
        self._detail.setVisible(False)
        layout.addWidget(self._detail)

        self._result_label = QLabel()
        self._result_label.setProperty("role", "toolCallResult")
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._result_label.setVisible(False)
        layout.addWidget(self._result_label)

        self._expanded = False
        self._has_notable_result = False

    def _toggle(self):
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._result_label.setVisible(self._expanded and self._has_notable_result)

    def set_result(self, result: str):
        self._has_notable_result = any(m in result.lower() for m in _FAILURE_MARKERS)
        if self._has_notable_result:
            self._result_label.setText(result)
            self._result_label.setVisible(self._expanded)
        self.setProperty("status", "error" if self._has_notable_result else "ok")
        self.style().unpolish(self)
        self.style().polish(self)


class _SendButton(QPushButton):
    """Send button that grows a spinning arc around its border while a
    request is in flight — a lightweight stand-in for a busy indicator that
    needs no extra widgets or icons."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._angle = 0
        self._loading = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def start_loading(self):
        self._loading = True
        self._angle = 0
        self._timer.start()
        self.setProperty("busy", True)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def stop_loading(self):
        self._loading = False
        self._timer.stop()
        self.setProperty("busy", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _tick(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._loading:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.palette().color(QPalette.Highlight))
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawArc(rect, -self._angle * 16, 100 * 16)


class ChatWidget(QFrame):
    """Minimal chat sidebar panel: scrollback of bubbles + input box + send
    button + a model picker backed by whatever Ollama already has pulled
    (plus any local GGUF files it can import on first use)."""

    message_submitted = Signal(str)
    model_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self.setProperty("role", "seriesBrowserPanel")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumWidth(280)

        self._assistant_bubble: _Bubble | None = None
        self._assistant_buffer = ""
        self._import_worker: CallableWorker | None = None
        self._user_bubbles: list[_Bubble] = []
        self._busy = False
        self._mic_worker: MicrophoneWorker | None = None
        self._transcription_worker: TranscriptionWorker | None = None
        self._f9_held = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel("Assistant")
        title.setProperty("role", "sidebarSectionTitle")
        outer.addWidget(title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(5, 8, 5, 8)
        self._content_layout.setSpacing(4)
        self._content_layout.addStretch(1)

        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask something…")
        self._input.returnPressed.connect(self._on_send_clicked)
        input_row.addWidget(self._input, 1)

        self._mic_btn = _SendButton("🎤")
        self._mic_btn.setProperty("role", "sidebarButton")
        if VOICE_AVAILABLE:
            self._mic_btn.setToolTip("Hold to speak (or hold F9)\nRight-click to select microphone")
            self._mic_btn.pressed.connect(self._on_mic_press)
            self._mic_btn.released.connect(self._on_mic_release)
            self._mic_btn.setContextMenuPolicy(Qt.CustomContextMenu)
            self._mic_btn.customContextMenuRequested.connect(self._show_mic_device_menu)
            QApplication.instance().installEventFilter(self)
        else:
            self._mic_btn.setToolTip("Voice input unavailable — install faster-whisper and sounddevice.")
            self._mic_btn.setEnabled(False)
        input_row.addWidget(self._mic_btn)

        self._send_btn = _SendButton("Send")
        self._send_btn.setProperty("role", "sidebarButton")
        self._send_btn.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self._send_btn)
        outer.addLayout(input_row)

        self._model_combo = QComboBox()
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        outer.addWidget(self._model_combo)

    # -- model picker ---------------------------------------------------

    def _populate_model_combo(self, select_tag: str | None = None):
        select_tag = select_tag or self.current_tag() or OLLAMA_MODEL
        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        pulled = list_ollama_models()
        pulled_tags = {m["tag"] for m in pulled}
        if not any(m["tag"] == OLLAMA_MODEL for m in pulled):
            pulled = [{"tag": OLLAMA_MODEL, "label": OLLAMA_MODEL, "local": False}] + pulled

        for m in pulled:
            self._model_combo.addItem(m["label"], m["tag"])

        for gguf_path in discover_local_gguf_files():
            tag = gguf_model_tag(gguf_path)
            if tag in pulled_tags:
                continue  # already imported — shown via its pulled entry above
            self._model_combo.addItem(f"{gguf_path.stem} (import)", str(gguf_path))

        if select_tag is not None:
            idx = self._model_combo.findData(select_tag)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)
        if self._model_combo.count():
            self.model_changed.emit(self._model_combo.currentData())

    def current_tag(self) -> str | None:
        return self._model_combo.currentData() if self._model_combo.count() else None

    def _on_model_selected(self, index: int):
        data = self._model_combo.itemData(index)
        if data is None:
            return
        if data.lower().endswith(".gguf"):
            self._start_import(data)
            return
        self.model_changed.emit(data)

    def _start_import(self, gguf_path_str: str):
        self.set_enabled(False)
        self._model_combo.setEnabled(False)
        self.append_system_message(f"Importing {Path(gguf_path_str).name} into Ollama…")
        self._send_btn.start_loading()

        self._import_worker = CallableWorker(import_gguf_model, Path(gguf_path_str))
        self._import_worker.finished_ok.connect(self._on_import_finished)
        self._import_worker.error.connect(self._on_import_error)
        # Clear the reference on `finished`, not `finished_ok`/`error` — those fire
        # from inside run() before the thread has actually stopped, so dropping the
        # last reference there races the thread teardown and aborts the process.
        self._import_worker.finished.connect(self._clear_import_worker)
        self._import_worker.start()

    def _clear_import_worker(self):
        self._import_worker = None

    def _finish_import(self, message: str, select_tag: str | None = None):
        self._send_btn.stop_loading()
        self.set_enabled(True)
        self._model_combo.setEnabled(True)
        self.append_system_message(message)
        self._populate_model_combo(select_tag=select_tag)

    def _on_import_finished(self, tag: str):
        self._finish_import(f"Model imported as {tag}.", select_tag=tag)

    def _on_import_error(self, message: str):
        self._finish_import(f"Model import failed: {message}")

    # -- chat input -------------------------------------------------------

    def _on_send_clicked(self):
        if self._busy:
            return  # keep typing while a request is in flight, just can't send yet
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._add_bubble("user", text)
        self.message_submitted.emit(text)

    def set_enabled(self, enabled: bool):
        self._busy = not enabled
        self._send_btn.setEnabled(enabled)
        if enabled:
            self._send_btn.stop_loading()
        else:
            self._send_btn.start_loading()

    # -- voice input (push-to-talk) ------------------------------------------

    def eventFilter(self, obj, event):
        """Global F9 push-to-talk, so it works from any focused widget."""
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key_F9 and not self._f9_held:
                self._f9_held = True
                self._on_mic_press()
                return True
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key_F9 and self._f9_held:
                self._f9_held = False
                self._on_mic_release()
                return True
        return super().eventFilter(obj, event)

    def _show_mic_device_menu(self, pos):
        """Right-click context menu on the mic button to choose an input device."""
        devices = list_input_devices()
        if not devices:
            return
        current = selected_input_device()
        menu = QMenu(self)
        act_default = menu.addAction("System default")
        act_default.setCheckable(True)
        act_default.setChecked(current is None)
        menu.addSeparator()
        actions = {}
        for idx, name in devices:
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(idx == current)
            actions[act] = idx

        chosen = menu.exec(self._mic_btn.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_default:
            select_input_device(None)
        elif chosen in actions:
            select_input_device(actions[chosen])

    def _on_mic_press(self):
        if not VOICE_AVAILABLE or self._mic_worker is not None:
            return
        # setDown() gives the mic button the same pressed accent look as the
        # send button while it's held — also covers the F9 path, which has
        # no real mouse press for Qt's own :pressed state to latch onto.
        self._mic_btn.setDown(True)
        self._mic_worker = MicrophoneWorker(self)
        self._mic_worker.finished.connect(self._on_mic_audio_ready)
        self._mic_worker.error.connect(self._on_mic_error)
        self._mic_worker.start()

    def _on_mic_release(self):
        if self._mic_worker is None:
            return
        self._mic_worker.stop()
        self._mic_worker = None
        self._mic_btn.setDown(False)
        self._mic_btn.start_loading()  # same spinning-arc busy indicator as Send

    def _on_mic_audio_ready(self, audio):
        self._transcription_worker = TranscriptionWorker(audio, parent=self)
        self._transcription_worker.finished.connect(self._on_transcription_done)
        self._transcription_worker.error.connect(self._on_mic_error)
        self._transcription_worker.start()

    def _on_transcription_done(self, text: str):
        self._mic_btn.stop_loading()
        self._transcription_worker = None
        if not text:
            return
        current = self._input.text()
        if current and not current.endswith(" "):
            text = " " + text
        self._input.insert(text)
        self._input.setFocus()

    def _on_mic_error(self, message: str):
        self._mic_btn.setDown(False)
        self._mic_btn.stop_loading()
        self._mic_worker = None
        self._transcription_worker = None
        self.append_system_message(f"Voice error: {message}")

    # -- bubbles ------------------------------------------------------------

    def _add_bubble(self, role: str, text: str) -> _Bubble:
        bubble = _Bubble(role)
        bubble.set_text(text)
        self._add_row(bubble, role)
        return bubble

    def _add_row(self, widget: QWidget, role: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            widget.apply_width_cap(self._user_bubble_max_width())
            self._user_bubbles.append(widget)
            row.addStretch(1)
            row.addWidget(widget)
        else:
            row.addWidget(widget, 1)  # stretch factor: fill the row width

        container = QWidget()
        container.setLayout(row)
        insert_pos = self._content_layout.count() - 1  # before trailing stretch
        self._content_layout.insertWidget(insert_pos, container)
        self._scroll_to_bottom()

    def _user_bubble_max_width(self) -> int:
        margins = self._content_layout.contentsMargins()
        available = self._scroll.viewport().width() - margins.left() - margins.right()
        return max(120, int(available * _USER_BUBBLE_WIDTH_FRACTION))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cap = self._user_bubble_max_width()
        for bubble in self._user_bubbles:
            bubble.apply_width_cap(cap)

    def _scroll_to_bottom(self):
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def begin_assistant_message(self):
        self._assistant_buffer = ""
        self._assistant_bubble = None  # created lazily on the first chunk, so a
        # response that goes straight to a tool call (no streamed text) never
        # leaves an empty placeholder bubble behind.

    def append_assistant_chunk(self, chunk: str):
        self._assistant_buffer += chunk
        if self._assistant_bubble is None:
            self._assistant_bubble = self._add_bubble("assistant", "")
        self._assistant_bubble.set_text(self._assistant_buffer)
        self._scroll_to_bottom()

    def finalize_assistant_message(self) -> str:
        text = self._assistant_buffer
        self._assistant_buffer = ""
        self._assistant_bubble = None
        return text

    def append_system_message(self, text: str):
        self._add_bubble("system", text)

    def append_tool_call(self, name: str, arguments: dict) -> _ToolCallBubble:
        bubble = _ToolCallBubble(name, arguments)
        self._add_row(bubble, "tool")
        return bubble
