"""QThread bases: one for any blocking call, one for the streaming chat transports."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class CallableWorker(QThread):
    finished_ok = Signal(object)  # whatever `func` returns
    error = Signal(object)  # the raised exception itself, so callers can inspect its type
    progress = Signal(object)  # whatever `func` passes to its `report` callback, with report=True

    def __init__(self, func, *args, report: bool = False, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = {**kwargs, "report": self.progress.emit} if report else kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(e)
            return
        self.finished_ok.emit(result)


class ChatWorker(QThread):
    """Base for the chat transports, which must stay drop-in for each other.

    ChatController asks llm_devices for a class and never learns which it holds, so the
    signals and constructor live here rather than being duplicated per wire format.
    """

    token_received = Signal(str)
    # Reasoning text; both formats keep it out of the reply, so the view can fold it away.
    thinking_received = Signal(str)
    tool_calls_ready = Signal(list)  # list of (tool_name, arguments) tuples
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, messages: list[dict], tools: list[dict] | None = None, model: str = "",
                 think: bool = False):
        super().__init__()
        self._messages = messages
        self._tools = tools
        self._model = model
        self._think = think
