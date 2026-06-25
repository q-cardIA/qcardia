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

    def load_lge_series(self, folder_path):
        try:
            volume, slice_positions = self.model.load_lge_series(folder_path)
            self.view.display_area_widget.set_lge_volume(
                volume, slice_positions=slice_positions
            )
        except Exception as e:
            logging.error(f"Failed to load LGE series: {e}")
            self.view.show_error(
                "LGE Load Error", f"Failed to load LGE series:\n{str(e)}"
            )

    def load_cine_series(self, folder_path):
        try:
            temporal_volumes, num_frames = self.model.load_cine_series(folder_path)

            self.current_time_index = 0
            self.view.update_display(
                temporal_volumes[self.current_time_index],
                slice_positions=self.model.slice_positions,
            )
            self.view.update_time_slider_range(num_frames)
        except Exception as e:
            logging.error(f"Failed to load cine series: {e}")
            self.view.show_error(
                "Load Error", f"Failed to load cine series:\n{str(e)}"
            )

    def run_segmentation(self):
        try:
            temporal_segmentations = self.model.predict_segmentation()

            if temporal_segmentations:
                idx = min(self.current_time_index, len(temporal_segmentations) - 1)
                self.view.display_area_widget.set_segmentation_volume(
                    temporal_segmentations[idx]
                )
        except Exception as e:
            logging.error(f"Segmentation failed: {e}")
            self.view.show_error(
                "Segmentation Error", f"Failed to run segmentation:\n{str(e)}"
            )

    def run_lge_segmentation(self):
        try:
            seg_volume = self.model.predict_lge_segmentation()
            self.view.display_area_widget.set_lge_segmentation_volume(seg_volume)
        except ValueError as e:
            self.view.show_warning("No LGE Volume", str(e))
        except Exception as e:
            logging.error(f"LGE segmentation failed: {e}")
            self.view.show_error(
                "LGE Segmentation Error", f"Failed to run LGE segmentation:\n{str(e)}"
            )

    def set_time_frame(self, time_index):
        temporal_volumes = self.model.temporal_volumes
        temporal_segmentations = self.model.temporal_segmentations

        if 0 <= time_index < len(temporal_volumes):
            self.current_time_index = time_index
            self.view.update_display_with_position(
                temporal_volumes[time_index],
                preserve_position=True,
                slice_positions=self.model.slice_positions,
            )

            if temporal_segmentations and time_index < len(temporal_segmentations):
                self.view.display_area_widget.set_segmentation_volume(
                    temporal_segmentations[time_index]
                )

    def save_segmentation(self):
        try:
            output_path = self.model.save_segmentation()
            logging.info(f"Segmentation saved to {output_path}")
            self.view.show_info(
                "Save Successful",
                f"Segmentation saved successfully to:\n{output_path}",
            )
        except ValueError as e:
            self.view.show_warning("No Segmentation", str(e))
        except Exception as e:
            logging.error(f'Failed to save segmentation: {e}"')
            self.view.show_error(
                "Save Failed", f"Failed to save segmentation:\n{str(e)}"
            )
