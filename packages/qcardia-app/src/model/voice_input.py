"""Push-to-talk voice input using faster-whisper for transcription.

"""
from __future__ import annotations

import logging
import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

VOICE_AVAILABLE = False
_import_error: str = ""

try:
    import sounddevice as sd          # noqa: F401
    from faster_whisper import WhisperModel  # noqa: F401
    VOICE_AVAILABLE = True
except ImportError as _e:
    _import_error = str(_e)

SAMPLE_RATE = 16_000   # Hz — Whisper's native sample rate
DEFAULT_MODEL_SIZE = "small"

# Biases Whisper's decoding toward clinical/viewer vocabulary it otherwise
# mishears (e.g. "CINE" as "sign"/"seen").
INITIAL_PROMPT = (
    "Cardiac MRI viewer terms: CINE, LGE, SAX, LAX, four chamber, three "
    "chamber, two chamber, perfusion, stress, rest, T1 mapping, T2 mapping. "
    "Apply the functional preset. Show the CINE SAX segmentation."
)

# Selected input device index — None means sounddevice's default device.
_selected_device: "int | None" = None


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every available input device."""
    if not VOICE_AVAILABLE:
        return []
    devices = sd.query_devices()
    return [(i, dev["name"]) for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]


def selected_input_device() -> "int | None":
    """Currently selected input device index, or None for system default."""
    return _selected_device


def select_input_device(index: "int | None"):
    """Set the microphone device to use. Pass None to revert to system default."""
    global _selected_device
    _selected_device = index


class MicrophoneWorker(QThread):
    """Records microphone audio until stop() is called.

    Emits finished(audio_array) where audio_array is float32 mono at SAMPLE_RATE.
    """

    finished = Signal(object)   # np.ndarray float32, shape (n_samples,)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        if not VOICE_AVAILABLE:
            self.error.emit(f"sounddevice not installed: {_import_error}")
            return

        chunks: list[np.ndarray] = []

        def _callback(indata, frames, time_info, status):
            if status:
                logging.debug("Mic: %s", status)
            chunks.append(indata.copy())

        self._stop_event.clear()
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                 callback=_callback, device=_selected_device):
                self._stop_event.wait()
        except Exception as exc:
            self.error.emit(str(exc))
            return

        audio = np.concatenate(chunks, axis=0).flatten() if chunks else np.zeros(0, dtype=np.float32)
        self.finished.emit(audio)


class TranscriptionWorker(QThread):
    """Runs faster-whisper on a float32 audio array in a background thread."""

    finished = Signal(str)   # transcribed text (may be empty)
    error = Signal(str)

    _model: "WhisperModel | None" = None
    _loaded_size: str = ""
    _lock = threading.Lock()

    def __init__(self, audio: np.ndarray, model_size: str = DEFAULT_MODEL_SIZE, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._model_size = model_size

    @classmethod
    def preload_model(cls, model_size: str = DEFAULT_MODEL_SIZE):
        """Download and cache the Whisper model (call once at app startup)."""
        if not VOICE_AVAILABLE:
            return
        cls._load(model_size)

    @classmethod
    def _load(cls, model_size: str):
        with cls._lock:
            if cls._model is None or cls._loaded_size != model_size:
                cls._model = WhisperModel(model_size, device="cpu", compute_type="int8")
                cls._loaded_size = model_size
                logging.info("Voice input ready — model: whisper-%s", model_size)

    def run(self):
        if not VOICE_AVAILABLE:
            self.error.emit(f"faster-whisper not installed: {_import_error}")
            return

        # Discard recordings shorter than 300 ms (likely accidental touches).
        if len(self._audio) < SAMPLE_RATE * 0.3:
            self.finished.emit("")
            return

        rms = float(np.sqrt(np.mean(self._audio ** 2)))
        if rms < 1e-6:
            self.error.emit("No audio captured — is the microphone muted?")
            return

        # Normalise to a target RMS so Whisper always receives well-levelled
        # audio, regardless of the OS microphone gain setting.
        _TARGET_RMS = 0.1
        audio = np.clip(self._audio * (_TARGET_RMS / rms), -1.0, 1.0)

        try:
            self._load(self._model_size)
            segments, _info = TranscriptionWorker._model.transcribe(
                audio,
                language="en",           # auto-detect
                beam_size=5,
                vad_filter=False,        # disabled: VAD strips push-to-talk clips
                no_speech_threshold=0.4, # default 0.6 — more permissive
                temperature=0.0,
                initial_prompt=INITIAL_PROMPT,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logging.info("Voice: %r", text)
            self.finished.emit(text)
        except Exception as exc:
            logging.error("Transcription error: %s", exc)
            self.error.emit(str(exc))
