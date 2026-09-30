"""Push-to-talk voice input using faster-whisper for transcription."""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import sys
import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

VOICE_AVAILABLE = False
_import_error: str = ""

try:
    import sounddevice as sd  # noqa: F401
    from faster_whisper import WhisperModel  # noqa: F401
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

    VOICE_AVAILABLE = True
except ImportError as _e:
    _import_error = str(_e)

SAMPLE_RATE = 16_000  # Hz — Whisper's native sample rate
BACKENDS = (("cuda", "float16", "large-v3-turbo"), ("cpu", "int8", "small"))
PARTIAL_STEP_S = 0.4
MIN_PARTIAL_S = 0.6
PAUSE_S = 1.0
MAX_PART_S = 25.0

logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Biases Whisper's decoding toward clinical/viewer vocabulary it otherwise
# mishears (e.g. "CINE" as "sign"/"seen").
INITIAL_PROMPT = (
    "Cardiac MRI viewer terms: CINE, LGE, EGE, SAX, LAX, four chamber, three chamber, two chamber, "
    "perfusion, stress, rest, ischaemia, T1 mapping, T2 mapping, viability, scar, fibrosis, infarct, "
    "transmurality, myocardium, left ventricle, right ventricle, ejection fraction, end-diastole, "
    "end-systole, segmentation, contours, functional preset, viability preset, perfusion preset. "
    "LGE is late gadolinium enhancement (late gad). EGE is early gadolinium enhancement. "
    "SAX is short axis. LAX is long axis."
)

# Selected input device index — None means sounddevice's default device.
_selected_device: "int | None" = None

_model: "WhisperModel | None" = None
_lock = threading.Lock()


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every available input device."""
    if not VOICE_AVAILABLE:
        return []
    devices = sd.query_devices()
    return [
        (i, dev["name"])
        for i, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
    ]


def selected_input_device() -> "int | None":
    """Currently selected input device index, or None for system default."""
    return _selected_device


def select_input_device(index: "int | None"):
    """Set the microphone device to use. Pass None to revert to system default."""
    global _selected_device
    _selected_device = index


def preload_model():
    """Download and cache the Whisper model (call once at app startup)."""
    if VOICE_AVAILABLE:
        _load()


def _add_cuda_libraries():
    spec = importlib.util.find_spec("nvidia")
    if sys.platform != "win32" or spec is None:
        return
    for folder in spec.submodule_search_locations:
        for bin_dir in glob.glob(os.path.join(folder, "*", "bin")):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _load() -> "WhisperModel":
    global _model
    with _lock:
        if _model is None:
            import ctranslate2

            _add_cuda_libraries()
            get_vad_model()
            errors = []
            for device, compute_type, size in BACKENDS:
                if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
                    continue
                try:
                    model = WhisperModel(size, device=device, compute_type=compute_type)
                    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
                except Exception as exc:
                    errors.append(f"{device}: {exc}")
                    continue
                _model = model
                logging.info("Voice input ready — model: whisper-%s on %s", size, device)
                break
            else:
                raise RuntimeError("No Whisper backend could be loaded: " + "; ".join(errors))
        return _model


def _transcribe(audio: np.ndarray, beam_size: int, context: str, max_new_tokens: int | None = None) -> str:
    model = _load()
    # Normalise to a target RMS so Whisper always receives well-levelled
    # audio, regardless of the OS microphone gain setting.
    _TARGET_RMS = 0.1
    rms = float(np.sqrt(np.mean(audio**2)))
    audio = np.clip(audio * (_TARGET_RMS / rms), -1.0, 1.0)
    with _lock:
        segments, _info = model.transcribe(
            audio,
            language="en",  # auto-detect
            beam_size=beam_size,
            vad_filter=False,  # disabled: VAD strips push-to-talk clips
            no_speech_threshold=0.4,  # default 0.6 — more permissive
            temperature=0.0,
            initial_prompt=f"{INITIAL_PROMPT} {context[-200:]}" if context else INITIAL_PROMPT,
            max_new_tokens=max_new_tokens,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def _speech(audio: np.ndarray) -> list[dict]:
    return get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=int(PAUSE_S * 1000), speech_pad_ms=200))


class MicrophoneWorker(QThread):
    """Records microphone audio until stop() is called, with live text, after whisper_streaming.

    While recording, the open part of the speech is decoded again every PARTIAL_STEP_S (greedy, leading
    silence cut, length capped so a half-spoken word cannot loop) and emitted as partial(stable, tentative):
    words two passes agree on, and the rest. A pause of PAUSE_S (Silero VAD), MAX_PART_S of speech, or
    stop() closes the part: it is decoded once with beam search, never changed again, and passed on as
    context. Emits finished(text) with the whole transcript (may be empty).
    """

    partial = Signal(str, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._fixed: list[str] = []

    def stop(self):
        self.requestInterruption()
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
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_callback,
                device=_selected_device,
            ):
                start = self._live_text(chunks)

            audio = (
                np.concatenate(chunks, axis=0).ravel()
                if chunks
                else np.zeros(0, dtype=np.float32)
            )
            if len(audio) and not self._fixed and float(np.sqrt(np.mean(audio**2))) < 1e-6:
                self.error.emit("No audio captured — is the microphone muted?")
                return
            tail = audio[start:]
            # Discard recordings shorter than 300 ms (likely accidental touches).
            if len(tail) >= SAMPLE_RATE * 0.3:
                speech = _speech(tail)
                if speech or not self._fixed:
                    begin, end = (speech[0]["start"], speech[-1]["end"]) if speech else (0, len(tail))
                    self._fix(tail[begin:end])
            text = " ".join(self._fixed)
            logging.info("Voice: %r", text)
            self.finished.emit(text)
        except Exception as exc:
            logging.error("Voice input error: %s", exc)
            self.partial.emit(" ".join(self._fixed), "")
            self.error.emit(str(exc))

    def _fix(self, audio: np.ndarray):
        text = _transcribe(audio, 5, " ".join(self._fixed))
        if text:
            self._fixed.append(text)

    def _live_text(self, chunks: list[np.ndarray]) -> int:
        start, decoded, previous = 0, 0, []
        while not self._stop_event.wait(0.05):
            pending = list(chunks)
            n = sum(len(c) for c in pending)
            if n - start < MIN_PARTIAL_S * SAMPLE_RATE or n - decoded < PARTIAL_STEP_S * SAMPLE_RATE:
                continue
            decoded = n
            part = np.concatenate(pending, axis=0).ravel()[start:]
            speech = _speech(part)
            if self._stop_event.is_set():
                break
            if not speech:
                start += max(0, len(part) - SAMPLE_RATE // 2)
                continue
            begin, end = speech[0]["start"], speech[-1]["end"]
            paused = len(part) - end >= 0.5 * PAUSE_S * SAMPLE_RATE
            if paused or len(part) >= MAX_PART_S * SAMPLE_RATE:
                cut = end if paused else len(part)
                self._fix(part[begin:cut])
                start += cut
                words = []
            else:
                limit = int(8 * (len(part) - begin) / SAMPLE_RATE) + 10
                words = _transcribe(part[begin:], 1, " ".join(self._fixed), limit).split()
            agreed = len(os.path.commonprefix([words, previous]))
            if not self._stop_event.is_set():
                self.partial.emit(" ".join([*self._fixed, *words[:agreed]]), " ".join(words[agreed:]))
            previous = words
        return start
