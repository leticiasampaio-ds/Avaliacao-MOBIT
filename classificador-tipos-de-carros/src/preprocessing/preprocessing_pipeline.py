import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.image_preprocessor import ImagePreprocessor

class PreprocessingPipeline:
    """
    Pipeline for image preprocessing:
    - Performs dataset split (train/val/test).
    - Applies resizing, normalization, augmentation and denoising.
    - Saves processed images in structured folders.

    Args:
        df_metadata_path (Path): Path to DataFrame with image paths and labels.
        output_dir (Path): Path to save processed images.
    """
    def __init__(self, df_metadata_path: Path, output_dir: Path):
        self.df_metadata = pd.read_csv(df_metadata_path)
        self.output_dir = output_dir

    def split_dataset(self):
        """
        Splits dataset into train, validation, and test sets.
        Returns:
            df_train, df_val, df_test (DataFrames)
        """
        df_train_val, df_test = train_test_split(
            self.df_metadata,
            test_size=0.2,
            stratify=self.df_metadata["label"],
            random_state=42
        )
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=0.25,
            stratify=df_train_val["label"],
            random_state=42
        )
        return df_train, df_val, df_test

    def run(self):
        """
        Executes the full preprocessing pipeline.
        """
        df_train, df_val, df_test = self.split_dataset()

        # Process and save each set
        ImagePreprocessor(df_train, os.path.join(self.output_dir,"train"), augmentations=True).preprocess_and_save()
        ImagePreprocessor(df_val, os.path.join(self.output_dir,"val"), augmentations=False).preprocess_and_save()
        ImagePreprocessor(df_test, os.path.join(self.output_dir,"test"), augmentations=False).preprocess_and_save()
