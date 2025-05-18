from pathlib import Path
import pandas as pd
from scipy.io import loadmat

class LoadMetadata:
    """
    Class responsible for loading and processing metadata from a .mat file 
    of the BMW-10 dataset.

    Args:
        mat_file_path (Path): Path to the .mat file containing image metadata.
    """
    def __init__(self, mat_file_path: Path):
        mat = loadmat(mat_file_path)
        self.anotation_raw = mat["annos"]
    
    def extract_metadata(self) -> list:
        """
        Extracts metadata from each image stored in the .mat file.

        Returns:
            list: A list of lists containing image filename, class ID, bounding box coordinates, 
                  and image dimensions.
        """
        metadata = []

        for item in self.anotation_raw[0]:
            fname = item[0]
            class_id = item[1]
            bbox_x1 = item[2]
            bbox_x2 = item[3]
            bbox_y1 = item[4]
            bbox_y2 = item[5]
            height = item[6]
            width = item[7]
            
            metadata.append([fname, class_id, bbox_x1, bbox_x2, bbox_y1, bbox_y2, height, width])
        
        return metadata

    def convert_list_to_dataframe(self, metadata_list: list) -> pd.DataFrame:
        """
        Converts the extracted metadata list into a pandas DataFrame.

        Args:
            metadata_list (list): List of metadata entries.

        Returns:
            pd.DataFrame: DataFrame with columns ['fname', 'class', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2', 'height', 'width'].
        """
        df_metadata = pd.DataFrame(metadata_list, columns=["fname", "class", "bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2", "height", "width"])
        return df_metadata
    
    def save_dataframe_to_csv(self, df_metadata: pd.DataFrame):
        """
        Saves dataframe with metadata to 'data/processed' directory.

        Args:
            metadata (pd.DataFrame): DataFrame with BMW-10 dataset.
        """
        df_metadata.to_csv('/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/processed/metadata.csv', index=False)
    
    def run(self):
        """
        Runs the complete metadata extraction, conversion, save pipeline.

        Returns:
            pd.DataFrame: Processed metadata as a pandas DataFrame.
        """
        metadata = self.extract_metadata()
        df_metadata = self.convert_list_to_dataframe(metadata)
        self.save_dataframe_to_csv(df_metadata)