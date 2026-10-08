from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QFontMetricsF, QPainter, QPalette, QPen, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from model.background_worker import CallableWorker
from utils import study_log
from model.voice_input import (
    VOICE_AVAILABLE,
    MicrophoneWorker,
    list_input_devices,
    select_input_device,
    selected_input_device,
)
from utils import bindings, llm_devices
from utils.app_config import (
    discover_local_gguf_files,
    gguf_model_tag,
    import_gguf_model,
)
from view import theme
from view.theme import thin_loading_bar

_USER_BUBBLE_WIDTH_FRACTION = 0.7  # user bubbles can grow up to 70% of the panel width

# Indent per bullet level; Qt's 40px default is a ~5-character jump, this is about two.
_LIST_INDENT = 18

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
    "list_subjects": lambda a: "List subjects",
    "load_subject": lambda a: f"Load subject {a.get('subject_id')}",
    "run_segmentation": lambda a: f"Run {a.get('modality')} segmentation",
    "apply_preset": lambda a: f"Apply {a.get('preset')} preset",
    "display_series": lambda a: (
        f"Show {a.get('series_key')} in panel {a.get('panel_idx')}"
        # "transverse" is the default axis, not worth echoing back.
        + (f" ({a['axis']})" if a.get("axis") not in (None, "", "transverse") else "")
        + (f", slice {a['slice']}" if a.get("slice") is not None else "")
        + (f", as panel {a['match_panel']}" if a.get("match_panel") is not None else "")
    ),
    "set_time_frame": lambda a: f"Jump to time frame {a.get('frame')}",
    "set_grid": lambda a: f"Set a {a.get('rows')}x{a.get('cols')} grid",
    "set_split_grid": lambda a: (f"Split: {a.get('left_rows')}x{a.get('left_cols')} left, "
                                 f"{a.get('right_rows')}x{a.get('right_cols')} right"),
    "zoom": lambda a: f"Zoom ({a.get('mode')})",
    "set_window": lambda a: (f"Window {a.get('series_key')}: {str(a.get('mode', '')).replace('_', ' ')}"
                             + (f" x{a['steps']}" if (a.get("steps") or 1) > 1 else "")
                             + (f" (panels {', '.join(map(str, a['panels']))})" if a.get("panels") else "")),
    "set_contours": lambda a: (f"{'Show' if a.get('shown') else 'Hide'} contours"
                               + (f" in panels {', '.join(map(str, a['panels']))}" if a.get("panels") else "")),
    "set_playback": lambda a: ("Play the cines" if a.get("playing") else "Stop the cines")
                              + (f" at {a['speed']:g}x" if a.get("speed") is not None else ""),
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


def _as_markdown(text: str) -> str:
    """Assistant text for QLabel's markdown renderer, with inline HTML neutralised."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


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
        # Only the assistant formats its text; user and system lines are shown verbatim.
        self._markdown = role == "assistant"
        # Explicit: QLabel's AutoText default sniffs its input and would render stray markup.
        self._label.setTextFormat(
            Qt.TextFormat.MarkdownText if self._markdown else Qt.TextFormat.PlainText
        )
        layout.addWidget(self._label)

    def set_text(self, text: str):
        self._label.setText(_as_markdown(text) if self._markdown else text)
        if self._markdown:
            self._compact_list_indent()

    def _compact_list_indent(self):
        """Shrink Qt's 40px list indent, via the document QLabel parents to itself (undocumented)."""
        for doc in self._label.findChildren(QTextDocument):
            if doc.indentWidth() != _LIST_INDENT:
                doc.setIndentWidth(_LIST_INDENT)

    def natural_width(self) -> int:
        """Width needed to render the current text on one line — QLabel's own
        sizeHint() is unreliable for this once word-wrap is on, so measure
        directly with font metrics instead."""
        # Rounded up plus spare: at an exact fit, word wrap moves the last word to a hidden line.
        advance = QFontMetricsF(self._label.font()).horizontalAdvance(self._label.text())
        return math.ceil(advance) + self._MARGINS + 4

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
        self._detail.setTextFormat(Qt.TextFormat.PlainText)  # arguments and paths, verbatim
        self._detail.setText(f"{name}({json.dumps(arguments)})")
        self._detail.setVisible(False)
        layout.addWidget(self._detail)

        self._result_label = QLabel()
        self._result_label.setProperty("role", "toolCallResult")
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._result_label.setTextFormat(Qt.TextFormat.PlainText)
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


class _ThinkingBubble(QFrame):
    """Collapsed-by-default reasoning, folded the way tool calls are."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "chatBubbleThinking")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)

        self._header = QPushButton("Thinking…")
        self._header.setProperty("role", "toolCallHeader")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        self._detail = QLabel()
        self._detail.setProperty("role", "thinkingDetail")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._detail.setTextFormat(Qt.TextFormat.PlainText)  # raw thought, not a formatted answer
        self._detail.setVisible(False)
        layout.addWidget(self._detail)

        self._expanded = False
        self._buffer = ""
        self._started = time.monotonic()

    def append(self, text: str):
        self._buffer += text
        self._detail.setText(self._buffer.strip())

    def seal(self):
        """Swap the live label for how long it took, so a finished turn reads as finished."""
        self._header.setText(f"Thought for {time.monotonic() - self._started:.1f}s")

    def _toggle(self):
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)


class _ChatInput(QPlainTextEdit):
    """Message box that grows with its text up to MAX_LINES; Enter sends, Shift+Enter adds a line."""

    submitted = Signal()
    MAX_LINES = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "chatInput")
        self.setTabChangesFocus(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().setDocumentMargin(2)
        self.document().documentLayout().documentSizeChanged.connect(self._fit)
        self._preview: QTextCursor | None = None
        self._preview_edited = False

    def text(self) -> str:
        return self.toPlainText()

    def insert(self, text: str):
        self.insertPlainText(text)

    def begin_preview(self):
        self._preview, self._preview_edited = QTextCursor(self.textCursor()), False
        self._preview.clearSelection()

    def show_preview(self, stable: str, tentative: str):
        if self._preview is None:
            return
        cursor = self._preview
        (cursor.joinPreviousEditBlock if self._preview_edited else cursor.beginEditBlock)()
        cursor.removeSelectedText()
        start = cursor.position()
        space = " " if (stable or tentative) and start and not self.document().characterAt(start - 1).isspace() else ""
        cursor.insertText(space + stable + (" " if stable and tentative else ""), QTextCharFormat())
        muted = QTextCharFormat()
        muted.setForeground(theme.color("text_muted"))
        cursor.insertText(tentative, muted)
        cursor.endEditBlock()
        self._preview_edited = True
        end = cursor.position()
        self.setTextCursor(cursor)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)

    def end_preview(self, text: str | None):
        if text is not None:
            self.show_preview(text, "")
        self._preview = None

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def _fit(self):
        # From the laid-out text: frameWidth() also counts the text padding.
        doc = self.document()
        layout = doc.documentLayout()
        text_height, block = 0.0, doc.firstBlock()
        while block.isValid():
            text_height += layout.blockBoundingRect(block).height()
            block = block.next()
        lines = max(1, round(doc.size().height()))
        per_line = text_height / lines
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded if lines > self.MAX_LINES else Qt.ScrollBarAlwaysOff)
        margins = self.contentsMargins()
        height = round(per_line * min(lines, self.MAX_LINES) + 2 * doc.documentMargin()) + margins.top() + margins.bottom()
        if height != self.height():
            self.setFixedHeight(height)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._fit()


_NAME_ACRONYMS = {"gpt": "GPT", "oss": "OSS", "llm": "LLM", "ai": "AI", "vl": "VL"}


def short_model_name(tag: str | None) -> str:
    """Ollama tag to a short display name: "deepseek-v4.1:latest" to "Deepseek V4.1", "qwen3:8b" to "Qwen 3 8B"."""
    if not tag:
        return "No model"
    base, _, variant = tag.partition(":")
    words = [w for w in re.split(r"[-_ ]+", re.sub(r"([A-Za-z]{2,})(\d)", r"\1 \2", base)) if w]
    if variant and variant != "latest":
        words.append(variant)
    pretty = []
    for w in words:
        if w.lower() in _NAME_ACRONYMS:
            pretty.append(_NAME_ACRONYMS[w.lower()])
        elif re.fullmatch(r"\d+(\.\d+)?[bm]", w.lower()):
            pretty.append(w.upper())
        else:
            pretty.append(w[:1].upper() + w[1:])
    return " ".join(pretty)


class _ModelPill(QPushButton):
    """Painted, so the chevron follows the (elided) name directly."""

    _PAD = 10
    _GAP = 6
    _CHEVRON = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = ""
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setMinimumWidth(60)
        self.setAttribute(Qt.WA_Hover)

    def set_label(self, label: str):
        self._label = label
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        text = math.ceil(QFontMetricsF(self.font()).horizontalAdvance(self._label))
        return QSize(2 * self._PAD + text + self._GAP + self._CHEVRON, max(24, self.fontMetrics().height() + 8))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        active = self.underMouse() or (self.menu() is not None and self.menu().isVisible())
        if active:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.color("bg3"))
            painter.drawRoundedRect(QRectF(self.rect()), self.height() / 2, self.height() / 2)
        metrics = QFontMetricsF(self.font())
        room = self.width() - 2 * self._PAD - self._GAP - self._CHEVRON
        text = metrics.elidedText(self._label, Qt.ElideRight, room)
        painter.setPen(theme.color("text" if active else "text_muted"))
        painter.drawText(QRect(self._PAD, 0, room, self.height()), Qt.AlignVCenter | Qt.AlignLeft, text)
        x = self._PAD + math.ceil(metrics.horizontalAdvance(text)) + self._GAP
        chevron = QRect(x, (self.height() - self._CHEVRON) // 2, self._CHEVRON, self._CHEVRON)
        theme.line_icon("chevron_down", "text_faint").paint(painter, chevron)


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
        self.update()

    def stop_loading(self):
        self._loading = False
        self._timer.stop()
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


_PTT_EVENTS = (QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.ShortcutOverride,
               QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick)


class ChatWidget(QFrame):
    """Minimal chat sidebar panel: scrollback of bubbles + input box + send
    button + a model picker backed by whatever Ollama already has pulled
    (plus any local GGUF files it can import on first use)."""

    message_submitted = Signal(str)
    model_changed = Signal(str)
    think_changed = Signal(bool)
    device_changed = Signal(object)  # the new Device, once it is actually serving

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumWidth(theme.scaled(280))

        self._assistant_bubble: _Bubble | None = None
        self._thinking_bubble: _ThinkingBubble | None = None
        self._assistant_buffer = ""
        self._import_worker: CallableWorker | None = None
        self._user_bubbles: list[_Bubble] = []
        self._busy = False
        self._busy_before_switch = False
        self._mic_worker: MicrophoneWorker | None = None
        self._ptt_held = False
        self._send_after_voice = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)


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

        self._composer = QFrame()
        self._composer.setObjectName("chatComposer")
        composer = QVBoxLayout(self._composer)
        composer.setContentsMargins(6, 12, 6, 6)  # the pill and the text start at the same x
        composer.setSpacing(4)
        self._input = _ChatInput()
        self._input.setPlaceholderText("Ask something…")
        self._input.submitted.connect(self._on_send_clicked)
        self._input.textChanged.connect(self._update_send_button)
        composer.addWidget(self._input)

        self._mic_btn = _SendButton("")
        self._mic_btn.setProperty("role", "composerIcon")
        if VOICE_AVAILABLE:
            hold = bindings.current().get("push_to_talk")
            self._mic_btn.setToolTip(f"Hold to speak{f' (or hold {bindings.label(hold)})' if hold else ''}"
                                     "\nRight-click to select microphone")
            self._mic_btn.pressed.connect(self._on_mic_press)
            self._mic_btn.released.connect(self._on_mic_release)
            self._mic_btn.setContextMenuPolicy(Qt.CustomContextMenu)
            self._mic_btn.customContextMenuRequested.connect(self._show_mic_device_menu)
            QApplication.instance().installEventFilter(self)
        else:
            self._mic_btn.setToolTip("Voice input unavailable — install faster-whisper and sounddevice.")
            self._mic_btn.setEnabled(False)

        self._send_btn = _SendButton("")
        self._send_btn.setProperty("role", "composerSend")
        self._send_btn.setToolTip("Send (Enter)")
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._send_btn.setEnabled(False)  # always there, so the mic never moves under a held mouse
        self._apply_composer_icons()
        theme.notifier.changed.connect(self._apply_composer_icons)

        # Hidden: they hold the state that the model button's menu shows and sets.
        # Switchable mid-session, so a remote device dropping doesn't end the conversation.
        self._device_combo = QComboBox(self)
        self._device_combo.hide()
        self._populate_device_combo()
        self._device_combo.currentIndexChanged.connect(self._on_device_selected)
        self._model_combo = QComboBox(self)
        self._model_combo.hide()
        self._think_action = QAction("Think before answering", self)
        self._think_action.setCheckable(True)
        self._think_action.toggled.connect(self.think_changed.emit)
        self._model_btn = _ModelPill()
        self._model_btn.setProperty("role", "modelPicker")
        self._model_btn.setCursor(Qt.PointingHandCursor)
        self._model_menu = theme.StickyMenu(self._model_btn)
        self._model_menu.setToolTipsVisible(True)
        self._model_menu.aboutToShow.connect(self._rebuild_model_menu)
        self._model_btn.setMenu(self._model_menu)
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        self._think_action.toggled.connect(lambda _: self._update_model_button())
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(4)
        bottom_row.addWidget(self._model_btn)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self._mic_btn)
        bottom_row.addWidget(self._send_btn)
        composer.addLayout(bottom_row)
        outer.addWidget(self._composer)
        self._device_bar = thin_loading_bar()
        outer.addWidget(self._device_bar)
        # Driven by ChatController, which knows when the model is actually loaded.
        self._model_bar = thin_loading_bar()
        outer.addWidget(self._model_bar)

    def set_model_loading(self, loading: bool):
        """Show/hide the hairline strip under the model picker."""
        self._model_bar.setVisible(loading)

    # -- device picker --------------------------------------------------

    def _populate_device_combo(self):
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        active = llm_devices.current()
        for device in llm_devices.load_devices():
            self._device_combo.addItem(device.label, device)
            if device.id == active.id:
                self._device_combo.setCurrentIndex(self._device_combo.count() - 1)
        self._device_combo.blockSignals(False)

    def _on_device_selected(self, index: int):
        device = self._device_combo.itemData(index)
        if device is None or device.id == llm_devices.current().id:
            return
        self._device_combo.setEnabled(False)
        self._device_bar.show()
        # A switch can start mid-request; on failure the old device and its request live on.
        self._busy_before_switch = self._busy
        self.set_enabled(False)
        self.append_system_message(f"Switching to {device.label}…")
        worker = CallableWorker(llm_devices.activate, device)
        worker.finished_ok.connect(lambda result, d=device: self._on_device_switched(d, result))
        worker.error.connect(lambda exc, d=device: self._on_device_switch_failed(d, exc))
        self._device_switch_worker = worker
        worker.start()

    def _on_device_switched(self, device, result):
        self._device_combo.setEnabled(True)
        self._device_bar.hide()
        if result.ok:
            # The controller re-enables input itself once it has let go of any request on
            # the old device; a tool sequence still mid-run must stay blocked.
            self.set_enabled(not self._busy_before_switch)
            self.device_changed.emit(device)
            self.append_system_message(f"Now using {device.label}: {result.detail}")
            self._populate_model_combo()  # each device serves its own models
        else:
            self.set_enabled(not self._busy_before_switch)
            self.append_system_message(
                f"{device.label} unavailable. {result.detail} "
                f"Staying on {llm_devices.current().label}."
            )
            self._populate_device_combo()  # snap the combo back to what is actually active

    def _on_device_switch_failed(self, device, exc: Exception):
        self._device_combo.setEnabled(True)
        self._device_bar.hide()
        self.set_enabled(not self._busy_before_switch)
        self.append_system_message(f"Could not switch to {device.label}: {exc}")
        self._populate_device_combo()

    # -- model picker ---------------------------------------------------

    def _populate_model_combo(self, select_tag: str | None = None, refresh: bool = False):
        # The device's own default: a hosted gateway shares no model names with local Ollama.
        fallback = llm_devices.default_model()
        select_tag = select_tag or self.current_tag() or fallback
        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        # Cached by the device's probe, so the GUI thread pays no HTTP timeout here.
        pulled = llm_devices.list_models(refresh) if llm_devices.chat_available() else []
        pulled_tags = {m["tag"] for m in pulled}
        pinned = llm_devices.current().model  # e.g. SPIKE's "best-live", which no model list shows
        if pinned and pinned not in pulled_tags:
            pulled = [{"tag": pinned, "label": pinned, "local": False}] + pulled

        for m in pulled:
            self._model_combo.addItem(m["label"], m["tag"])

        # GGUF import runs this machine's ollama CLI on local paths, so it is local-only.
        if llm_devices.current().is_local:
            for gguf_path in discover_local_gguf_files():
                tag = gguf_model_tag(gguf_path)
                if tag in pulled_tags:
                    continue  # already imported — shown via its pulled entry above
                self._model_combo.addItem(f"{gguf_path.stem} (import)", str(gguf_path))

        if select_tag is not None:
            idx = self._model_combo.findData(select_tag)
            if idx < 0:
                # Carried-over tag isn't served here; index 0 would silently load whatever sorts first.
                idx = self._model_combo.findData(fallback)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)
        self._sync_think_toggle()
        self._update_model_button()
        if self._model_menu.isVisible():
            QTimer.singleShot(0, self._rebuild_model_menu)  # not from inside the menu's own click
        if self._model_combo.count():  # the default model if installed, else the first one
            self.model_changed.emit(self._model_combo.currentData())
        elif llm_devices.chat_available():
            self.append_system_message(f"No models installed. Run `ollama pull {llm_devices.default_model()}`.")

    def _sync_think_toggle(self):
        """Only offer Think where the model can do it; Ollama rejects the request otherwise."""
        if not llm_devices.chat_available():
            supported, reason = False, "Select a working device first"
        else:
            supported = llm_devices.supports_thinking(self.current_tag())
            reason = "Ask the model to reason before answering" if supported else "This model cannot reason"
        if not supported:
            self._think_action.setChecked(False)
        self._think_action.setEnabled(supported)
        self._think_action.setToolTip(reason)

    def _update_model_button(self):
        name = short_model_name(self.current_tag())
        self._model_btn.set_label(f"{name} · Think" if self.think_enabled() else name)
        self._model_btn.setToolTip(f"{self.current_tag() or 'No model'} on {llm_devices.current().label}"
                                   f", thinking {'on' if self.think_enabled() else 'off'}")

    def _update_send_button(self):
        """Accent only when there is text to send; muted while empty or while a request runs."""
        self._send_btn.setEnabled(not self._busy and bool(self._input.text().strip()))

    def _apply_composer_icons(self):
        for button, kind, token in ((self._mic_btn, "mic", "text_muted"), (self._send_btn, "arrow_up", "on_accent")):
            button.setIcon(theme.line_icon(kind, token))
            button.setIconSize(QSize(16, 16))

    def _rebuild_model_menu(self):
        menu = self._model_menu
        # Rebuilt while open after a device switch: keep the bottom edge (Think, devices) in place.
        anchor = menu.geometry().bottom() if menu.isVisible() else None
        menu.clear()
        self._add_choices(menu, "Model", self._model_combo)
        if self._model_combo.count() == 0:
            menu.addAction("No models installed").setEnabled(False)
        menu.addAction("Refresh model list", lambda: self._populate_model_combo(refresh=True))
        menu.addSeparator()
        menu.addAction(self._think_action)
        menu.addSeparator()
        self._add_choices(menu, "Device", self._device_combo)
        pill_top = self._model_btn.mapToGlobal(self._model_btn.rect().topLeft()).y()
        if anchor is not None and anchor <= pill_top + 2:  # opened upward
            menu.adjustSize()
            menu.move(menu.x(), anchor - menu.height() + 1)

    @staticmethod
    def _add_choices(menu, title: str, combo: QComboBox):
        """The combo's items as one exclusive group; picking one sets the combo."""
        menu.addAction(title).setEnabled(False)
        group = QActionGroup(menu)
        for i in range(combo.count()):
            action = menu.addAction(combo.itemText(i))
            group.addAction(action)
            action.setCheckable(True)
            action.setChecked(i == combo.currentIndex())
            action.setEnabled(combo.isEnabled())
            action.triggered.connect(lambda _=False, i=i: combo.setCurrentIndex(i))

    def think_enabled(self) -> bool:
        return self._think_action.isChecked() and self._think_action.isEnabled()

    def current_tag(self) -> str | None:
        return self._model_combo.currentData() if self._model_combo.count() else None

    def _on_model_selected(self, index: int):
        data = self._model_combo.itemData(index)
        if data is None:
            return
        if data.lower().endswith(".gguf"):
            self._start_import(data)
            return
        # Thinking is per-model, so re-check on every change, not only on rebuild.
        self._sync_think_toggle()
        self._update_model_button()
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
        self._populate_model_combo(select_tag=select_tag, refresh=True)

    def _on_import_finished(self, tag: str):
        self._finish_import(f"Model imported as {tag}.", select_tag=tag)

    def _on_import_error(self, message: str):
        self._finish_import(f"Model import failed: {message}")

    # -- chat input -------------------------------------------------------

    def _on_send_clicked(self):
        if self._busy:
            return  # keep typing while a request is in flight, just can't send yet
        if self._mic_worker is not None:
            # The worker still writes the whole transcript into the input, so sending now
            # would leave it to reappear in front of the next message.
            self._send_after_voice = True
            self._ptt_held = False
            self._on_mic_release()
            return
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._add_bubble("user", text)
        self.message_submitted.emit(text)

    def set_enabled(self, enabled: bool):
        self._busy = not enabled
        if enabled:
            self._send_btn.stop_loading()
        else:
            self._send_btn.start_loading()
        self._update_send_button()

    # -- voice input (push-to-talk) ------------------------------------------

    def eventFilter(self, obj, event):
        """Global push-to-talk on its binding, so it works from any focused widget."""
        kind = event.type()
        if kind == QEvent.Type.ApplicationDeactivate and self._ptt_held:
            self._end_hold()
        elif kind in _PTT_EVENTS and (self._ptt_held or QApplication.activeModalWidget() is None):
            ptt = bindings.current().get("push_to_talk")
            mouse = kind in _PTT_EVENTS[3:]
            release = kind in (QEvent.Type.KeyRelease, QEvent.Type.MouseButtonRelease)
            repeat = not mouse and event.isAutoRepeat()
            click = bindings.click_of(bindings.drag_binding(event.button(), event.modifiers())) if mouse else None
            if not ptt or (mouse and not click):
                pass
            elif self._ptt_held and bindings.last_key(click or bindings.hold_key(event)) == bindings.last_key(ptt):
                if release and not repeat:
                    self._end_hold()
                event.accept()
                return True
            elif not release and not repeat and (click or bindings.hold_binding(event)) == ptt:
                if kind != QEvent.Type.ShortcutOverride:
                    self._ptt_held = True
                    self._on_mic_press()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _end_hold(self):
        self._ptt_held = False
        self._on_mic_release()

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
        # send button while it's held — also covers the push-to-talk key, which has
        # no real mouse press for Qt's own :pressed state to latch onto.
        self._mic_btn.setDown(True)
        self._voice_span = study_log.Span("voice")
        self._input.begin_preview()
        self._input.setPlaceholderText("Listening…")  # without live text, nothing else shows it records
        self._mic_worker = MicrophoneWorker(self)
        self._mic_worker.partial.connect(self._input.show_preview)
        self._mic_worker.finished.connect(self._on_transcription_done)
        self._mic_worker.error.connect(self._on_mic_error)
        self._mic_worker.start()

    def _on_mic_release(self):
        if self._mic_worker is None or self._mic_worker.isInterruptionRequested():
            return
        self._mic_worker.stop()
        self._voice_span.end()
        self._voice_span = study_log.Span("transcription")
        self._mic_btn.setDown(False)
        self._mic_btn.start_loading()  # same spinning-arc busy indicator as Send

    def _on_transcription_done(self, text: str):
        self._voice_span.end(text=text)
        self._mic_btn.stop_loading()
        self._mic_worker = None
        self._input.setPlaceholderText("Ask something…")
        self._input.end_preview(text)
        self._input.setFocus()
        if self._send_after_voice:
            self._send_after_voice = False
            self._on_send_clicked()

    def _on_mic_error(self, message: str):
        self._voice_span.end(error=message)
        self._send_after_voice = False
        self._mic_btn.setDown(False)
        self._mic_btn.stop_loading()
        self._mic_worker = None
        self._input.setPlaceholderText("Ask something…")
        self._input.end_preview(None)
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
        # A turn ending in a tool call is never finalized, so close its reasoning here.
        self._seal_thinking()

    def append_thinking_chunk(self, chunk: str):
        if self._thinking_bubble is None:
            self._thinking_bubble = _ThinkingBubble()
            self._add_row(self._thinking_bubble, "thinking")
        self._thinking_bubble.append(chunk)
        self._scroll_to_bottom()

    def append_assistant_chunk(self, chunk: str):
        self._assistant_buffer += chunk
        # Models pad replies with trailing newlines, which would draw an empty bubble.
        text = self._assistant_buffer.strip()
        if not text:
            return
        if self._assistant_bubble is None:
            self._assistant_bubble = self._add_bubble("assistant", "")
        self._assistant_bubble.set_text(text)
        self._scroll_to_bottom()

    def _seal_thinking(self):
        if self._thinking_bubble is not None:
            self._thinking_bubble.seal()
            self._thinking_bubble = None

    def finalize_assistant_message(self) -> str:
        self._seal_thinking()
        # Stripped, so a whitespace-only reply never reaches history.
        text = self._assistant_buffer.strip()
        self._assistant_buffer = ""
        self._assistant_bubble = None
        return text

    def append_system_message(self, text: str):
        self._add_bubble("system", text)

    def append_tool_call(self, name: str, arguments: dict) -> _ToolCallBubble:
        bubble = _ToolCallBubble(name, arguments)
        self._add_row(bubble, "tool")
        return bubble
