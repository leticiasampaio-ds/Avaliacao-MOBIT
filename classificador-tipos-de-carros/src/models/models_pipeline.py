from pathlib import Path

from models.data_loader_builder import DataLoaderBuilder
from models.model_trainer import ModelTrainer

class ModelsPipeline:
    """
    Pipeline for initializing and managing the training of a deep learning model
    for image classification using preprocessed car dataset.

    Responsibilities:
    - Load training, validation, and test sets.
    - Initialize the model (ResNet18).
    - Prepare for training and evaluation steps.
    """

    def __init__(self, data_root: Path, device: str = "cpu"):
        self.device = device
        self.data_root = data_root
        self.batch_size = 64
        self.num_classes = 4  # 3, 4, 5, 'outros'

    def build_dataloaders(self):
        """
        Builds DataLoaders for train, val, and test sets using preprocessed images.
        """
        builder = DataLoaderBuilder(self.data_root, batch_size=self.batch_size)

        self.train_loader = builder.build_loader("train")
        self.val_loader = builder.build_loader("val")
        self.test_loader = builder.build_loader("test")

    def build_model(self):
        """
        Loads a ResNet18 model and adapts the final layer to the number of classes.
        """
        ModelTrainer(self.train_loader,self.val_loader,self.test_loader).run()

    def run(self):
        """
        Placeholder for running the full training and evaluation pipeline.
        This is where training and evaluation routines will be plugged in.
        """
        self.build_dataloaders()
        self.build_model()