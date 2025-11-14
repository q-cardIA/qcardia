import os

import pydicom


class DICOMModel:
    def __init__(self):
        self.dicom_files = []

    def load_dicom_files(self, folder_path):
        self.dicom_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    file_path = os.path.join(root, file)
                    try:
                        # dicom_data = pydicom.dcmread(file_path)
                        dicom_data = pydicom.read_file(file_path)
                        self.dicom_files.append(dicom_data)
                    except Exception as e:
                        print(f"Error reading DICOM file '{file_path}': {e}")
