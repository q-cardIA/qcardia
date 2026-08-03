"""PatientLoadWorker — runs DICOMModel.load_patient() off the GUI thread.

Cardisort classification + series loading can take tens of seconds; without
this the whole app freezes (including the chat panel) for that whole time.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class PatientLoadWorker(QThread):
    finished_ok = Signal(object)  # PatientLoadResult
    error = Signal(str)

    def __init__(self, model, folder_path):
        super().__init__()
        self._model = model
        self._folder_path = folder_path

    def run(self):
        try:
            result = self._model.load_patient(self._folder_path)
        except Exception as e:
            self.error.emit(str(e))
            return
        self.finished_ok.emit(result)
