import os
import pydicom
from collections import defaultdict

class DICOMSeriesExtractor:
    def __init__(self, desired_series=None, ignore_terms=None):
        self.desired_series = {series.upper().strip() for series in desired_series} if desired_series else set()
        self.ignore_terms = {term.upper().strip() for term in ignore_terms} if ignore_terms else set()
        self.matched_dicom_files = defaultdict(list)

    def process_dicom_file(self, dicom_path):
        """Process a single DICOM file to extract the series description."""
        try:
            dicom = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            series_description = getattr(dicom, 'SeriesDescription', 'Unknown').upper().strip() # This may be a cause of the problem as the DICOM files may not have the "SeriesDescription" attribution
            if any(ignore_term in series_description for ignore_term in self.ignore_terms): # This may also be a cause of the problem as no description will be returned if it contains an element in ignore_terms
                # Debug print can be enabled by uncommenting the next line
                # print(f"Ignore term found in series description '{series_description}' for file '{dicom_path}'")
                return None
            return dicom_path, series_description
        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
            return None

    def validate_dicom_series(self, base_path):
        """Validate DICOM series across all patients in the specified directory."""
        print("Starting validation of DICOM series...")
        for patient_dir in os.listdir(base_path):
            patient_path = os.path.join(base_path, patient_dir)
            if os.path.isdir(patient_path):
                print(f"Processing patient directory: {patient_dir}")
                series_found = set()
                all_series_descriptions = set()
                dicom_files = [os.path.join(root, file)
                               for root, _, files in os.walk(patient_path)
                               for file in files if file.lower().endswith('.dcm')]
                for dicom_file in dicom_files:
                    result = self.process_dicom_file(dicom_file)
                    if result:
                        dicom_path, series_description = result
                        all_series_descriptions.add(series_description)  # Store all descriptions as processed
                        if series_description in self.desired_series:  # Remove the lower() function call since the contents of both variables were already converted to uppercase earlier in the code
                            series_found.add(series_description)
                            match_found = True
                            if patient_dir not in self.matched_dicom_files:
                                self.matched_dicom_files[patient_dir] = []
                            self.matched_dicom_files[patient_dir].append(dicom_path)
                if not series_found:
                  print(f"Validated series for {patient_dir}: {series_found}")
                else:
                  print(f"Error: No desired series identified for patient '{patient_dir}'.")
                  print(f"Available series for patient '{patient_dir}': {all_series_descriptions}")  # Moved this line to only print the series descriptions if a series is not found
        print("Validation complete.")

# Usage example:
extractor = DICOMSeriesExtractor(desired_series=["SERIES1", "SERIES2"], ignore_terms=["IGNORE"])
extractor.validate_dicom_series("/path/to/dicom/files")
