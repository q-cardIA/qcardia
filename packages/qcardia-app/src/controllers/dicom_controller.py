import logging

from model.dicom_model import read_disk_thumbnail
from utils.hanging_protocols import PRESETS, build_perfusion_preset
from view.series_browser_widget import SeriesCardInfo


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0

        self.view.load_patient_button_signal.connect(self.load_patient)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.lge_seg_button_signal.connect(self.run_lge_segmentation)
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)
        self.view.preset_requested_signal.connect(self.apply_preset)

    @property
    def _widget(self):
        return self.view.display_area_widget

    def load_patient(self, folder_path):
        try:
            result = self.model.load_patient(folder_path)
        except Exception as e:
            logging.error(f"Failed to load patient: {e}")
            self.view.show_error("Load Error", f"Failed to load patient folder:\n{str(e)}")
            return

        if not result.entries:
            self.view.show_warning(
                "No Series Found", "No sequence directories found in this folder."
            )
            return

        max_frames = 0
        for key, info in result.loaded.items():
            self._widget.register_series(
                key, info["volume"], is_temporal=info["is_temporal"], geometry=info["geometry"]
            )
            if info["is_temporal"]:
                max_frames = max(max_frames, info["num_frames"])

        self.view.series_browser_widget.populate(self._build_browser_cards(result))

        if max_frames:
            self.current_time_index = 0
            self.view.update_time_slider_range(max_frames)

        if not result.loaded:
            self.view.show_warning(
                "No Loadable Series",
                "Cardisort did not recognize any loadable series in this folder; "
                "showing all detected sequences in the browser.",
            )

    def _build_browser_cards(self, result) -> list[SeriesCardInfo]:
        cards = []
        for entry in result.entries:
            is_lge_like = entry.sequence_name in {"DBLGE", "WBLGE"}
            loaded_here = entry.key is not None and entry.is_primary and entry.key in result.loaded

            thumbnail = (
                self._widget.get_thumbnail(entry.key)
                if loaded_here
                else read_disk_thumbnail(entry.path, is_lge_like)
            )

            if entry.key is not None:
                group = entry.key.split("_", 1)[0]
                label = entry.key if entry.is_primary else entry.path.name
            elif entry.sequence_name is not None:
                group = entry.sequence_name
                label = f"{entry.sequence_name}_{entry.plane_name}"
            else:
                group = "Unclassified"
                label = entry.path.name

            cards.append(
                SeriesCardInfo(
                    label=label,
                    group=group,
                    thumbnail=thumbnail,
                    draggable=loaded_here,
                    series_key=entry.key if loaded_here else None,
                )
            )
        return cards

    def run_segmentation(self):
        try:
            temporal_segmentations = self.model.predict_segmentation()
            if temporal_segmentations:
                import numpy as np
                seg_4d = np.stack(temporal_segmentations, axis=0)
                self._widget.set_segmentation("CINE_SAX", seg_4d)
        except Exception as e:
            logging.error(f"Segmentation failed: {e}")
            self.view.show_error("Segmentation Error", f"Failed to run segmentation:\n{str(e)}")

    def run_lge_segmentation(self):
        try:
            seg_volume = self.model.predict_lge_segmentation()
            self._widget.set_segmentation("LGE_SAX", seg_volume)
        except ValueError as e:
            self.view.show_warning("No LGE Volume", str(e))
        except Exception as e:
            logging.error(f"LGE segmentation failed: {e}")
            self.view.show_error("LGE Segmentation Error", f"Failed to run LGE segmentation:\n{str(e)}")

    def apply_preset(self, name: str):
        if name == "perfusion":
            avail = set(self.model.series.keys())
            preset = build_perfusion_preset(
                has_stress="PERF_STRESS_SAX" in avail,
                has_rest="PERF_REST_SAX" in avail,
            )
        else:
            preset = PRESETS.get(name)

        if preset is None:
            logging.warning(f"apply_preset: unknown preset {name!r}")
            return

        rows, cols = preset.grid
        self._widget.set_grid_size(rows, cols)
        for panel in preset.panels:
            if self._widget.is_series_registered(panel.key):
                self._widget.set_panel_series(panel.idx, panel.key, axis=panel.axis)
            else:
                self._widget.set_panel_empty(panel.idx)

    def set_time_frame(self, time_index):
        self.current_time_index = time_index
        self._widget.set_time_index(time_index)

    def save_segmentation(self):
        try:
            output_path = self.model.save_segmentation()
            self.view.show_info("Saved", f"Segmentation saved to:\n{output_path}")
        except Exception as e:
            logging.error(f"Save failed: {e}")
            self.view.show_error("Save Error", f"Failed to save segmentation:\n{str(e)}")
