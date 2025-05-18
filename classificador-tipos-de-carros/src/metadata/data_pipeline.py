from metadata.load_metadata import LoadMetadata
from metadata.standardize_data import StandardizeMetadata

from pathlib import Path

class DataPipeline:
    """
    Pipeline to load and standardize image metadata for car classification tasks.

    This class handles the integration between loading raw annotation data from a .mat file
    and preprocessing it to produce a clean and standardized DataFrame, including class filtering
    and path generation.

    Args:
        mat_file_path (Path): Path to the .mat file containing image annotations.
        images_root (Path): Path to the root directory containing the image files.
    """
    def __init__(self, mat_file_path: Path, images_root: Path):
        self.df_metadata = LoadMetadata(mat_file_path).run()
        self.images_root = images_root

    def run(self):
        """
        Executes the metadata standardization process, including:
        - Extracting relevant metadata fields
        - Cleaning and formatting annotations
        - Filtering target classes and mapping others to "Others"
        - Constructing absolute image paths

        Returns:
            pd.DataFrame: Standardized metadata ready for dataset splitting and model training.
        """
        return StandardizeMetadata(self.df_metadata, self.images_root).run()