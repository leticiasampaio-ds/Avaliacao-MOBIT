from data.load_metadata import LoadMetadata
from data.standardize_data import StandardizeData

from pathlib import Path

class DataPipeline:
    """
    Pipeline to preprocess metadata and prepare test/train data.

    Args:
        mat_file_path (Path): Path to the .mat file containing image metadata.
        images_root (Path): Root path to the directory where images are stored.
    """
    def __init__(self, mat_file_path: Path, images_root: Path):
        self.df_metadata = LoadMetadata(mat_file_path).run()
        self.images_root = images_root
    
    def run(self):
        """
        Runs ...
        """
        StandardizeData(self.df_metadata, self.images_root).run()