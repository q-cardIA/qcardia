import logging


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0

        self.view.load_button_signal.connect(self.load_cine_series)
        self.view.load_lge_button_signal.connect(self.load_lge_series)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.lge_seg_button_signal.connect(self.run_lge_segmentation)
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)

    @property
    def _widget(self):
        return self.view.display_area_widget

    def load_cine_series(self, folder_path):
        try:
            temporal_volumes, num_frames = self.model.load_cine_series(folder_path)
            geom = self.model.get_cine_geometry()
            volume_4d = temporal_volumes  # list[np.ndarray (Z,H,W)] → stack to (T,Z,H,W)
            import numpy as np
            volume_4d = np.stack(temporal_volumes, axis=0)
            self._widget.register_series("CINE_SAX", volume_4d, is_temporal=True, geometry=geom)
            self._widget.set_panel_series(0, "CINE_SAX", axis="transverse")
            self.current_time_index = 0
            self.view.update_time_slider_range(num_frames)
        except Exception as e:
            logging.error(f"Failed to load cine series: {e}")
            self.view.show_error("Load Error", f"Failed to load cine series:\n{str(e)}")

    def load_lge_series(self, folder_path):
        try:
            volume, _slice_positions = self.model.load_lge_series(folder_path)
            geom = self.model.get_lge_geometry()
            self._widget.register_series("LGE_SAX", volume, is_temporal=False, geometry=geom)
            self._widget.set_panel_series(1, "LGE_SAX", axis="transverse")
        except Exception as e:
            logging.error(f"Failed to load LGE series: {e}")
            self.view.show_error("LGE Load Error", f"Failed to load LGE series:\n{str(e)}")

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
