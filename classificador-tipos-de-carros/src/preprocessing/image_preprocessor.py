import cv2
import os
from pathlib import Path
import numpy as np
import albumentations as A
from tqdm import tqdm

class ImagePreprocessor:
    """
    Preprocesses and saves images with resizing, color conversion, normalization,
    optional data augmentation and denoising.

    Args:
        df_metadata (pd.DataFrame): DataFrame with image paths and labels.
        save_root (Path): Destination folder to save processed images.
        image_size (tuple): Desired output size (height, width).
        augmentations (bool): Whether to apply data augmentation (only for training data).
        denoising (bool): Whether to apply denoising.
    """
    def __init__(self, df_metadata, save_root, image_size=(224, 224), augmentations=False, denoising=False):
        self.df_metadata = df_metadata
        self.save_root = Path(save_root)
        self.image_size = image_size
        self.augmentations = augmentations
        self.denoising = denoising
        self.augmentor = self.build_augmentor() if augmentations else None

    def build_augmentor(self):
        """
        Constructs the augmentation pipeline using albumentations.

        Returns:
            A.Compose: A set of augmentation transformations.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3)
        ])

    def apply_denoising(self, image):
        """
        Applies histogram equalization to improve image contrast in YUV space.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Denoised RGB image.
        """
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def preprocess_and_save(self):
        """
        Applies resizing, color conversion, normalization, denoising and
        optional data augmentation to each image. Saves preprocessed images
        into folders based on their class labels.
        """
        for _, row in tqdm(self.df_metadata.iterrows(), total=len(self.df_metadata), desc="Processing Images"):
            img_path = row["image_path"]
            label = row["label"]

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)

            if self.denoising:
                image = self.apply_denoising(image)

            if self.augmentor:
                image = self.augmentor(image=image)["image"]

            image = image.astype(np.float32) / 255.0

            self.save_image(image, label, img_path)

    def save_image(self, image, label, original_path):
        """
        Saves a preprocessed image to its corresponding label directory.

        Args:
            image (np.ndarray): Normalized image to save.
            label (str or int): Image class label.
            original_path (Path): Original image path to preserve filename.
        """
        label_folder = os.path.join(self.save_root, str(label))
        os.makedirs(label_folder, exist_ok=True)  # make sure directory exists

        filename = Path(original_path).name
        save_path = os.path.join(label_folder, filename)

        image_to_save = (image * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))