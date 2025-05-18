import os
from pathlib import Path
import pandas as pd
import numpy as np

class StandardizeData:
    """
    Standardizes metadata for image classification tasks.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing raw metadata extracted from .mat file.
        images_root (Path): Root path to the directory where images are stored.
    """
    def __init__(self, df_metadata: pd.DataFrame, images_root: Path):
        self.df_metadata = df_metadata
        self.images_root = images_root

    def select_only_desired_columns(self):
        """
        Selects only the columns needed for classification: 'fname' and 'class'.
        """
        self.df_metadata = self.df_metadata[["fname", "class"]]

    def extract_values_from_list(self):
        """
        Extracts actual values from nested np.ndarrays or lists in 'fname' and 'class' columns.
        Also maps non-target classes to the label 'Others'.
        """
        self.df_metadata["fname"] = self.df_metadata["fname"].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
        )
        self.df_metadata["class"] = self.df_metadata["class"].apply(
            lambda x: int(x[0][0]) if isinstance(x, (list, np.ndarray)) and isinstance(x[0], (list, np.ndarray)) else int(x)
        )
        self.df_metadata["label"] = self.df_metadata["class"].apply(
            lambda x: x if x in [3, 4, 5] else "Others"
        )


    def build_image_paths(self):
        """
        Builds full image paths using the root directory and relative file names.
        Uses os.path.join for better cross-platform compatibility.
        """
        self.df_metadata["image_path"] = self.df_metadata["fname"].apply(
            lambda x: os.path.join(self.images_root, x)
        )

    def save_dataframe_to_csv(self, df_metadata: pd.DataFrame):
        """
        Saves dataframe with metadata to 'data/processed' directory.

        Args:
            metadata (pd.DataFrame): DataFrame with BMW-10 dataset.
        """
        df_metadata.to_csv('/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/processed/metadata.csv', index=False)

    def run(self) -> pd.DataFrame:
        """
        Executes the standardization pipeline and returns the processed DataFrame.

        Returns:
            pd.DataFrame: A standardized DataFrame with columns ['image_path', 'label'].
        """
        self.select_only_desired_columns()
        self.extract_values_from_list()
        self.build_image_paths()
        self.save_dataframe_to_csv(self.df_metadata[["image_path", "label"]])