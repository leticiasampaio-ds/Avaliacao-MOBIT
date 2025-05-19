import os
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

class DataLoaderBuilder:
    """
    Builds PyTorch DataLoaders for image classification tasks, supporting class balancing via WeightedRandomSampler.

    Args:
        data_root (Path): Root directory where preprocessed images are stored in subfolders per class.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
    """
    def __init__(self, data_root: Path, batch_size: int = 16, num_workers: int = 0):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def build_loader(self, subset: str) -> DataLoader:
        """
        Builds a DataLoader for a specified subset ('train', 'val', 'test').

        Args:
            subset (str): Dataset split to load.

        Returns:
            DataLoader: Configured PyTorch DataLoader.
        """
        subset_path = self.data_root / subset
        dataset = datasets.ImageFolder(root=subset_path, transform=self.transform)

        if subset == "train":
            # Compute class weights for oversampling
            class_counts = Counter([label for _, label in dataset.imgs])
            total = sum(class_counts.values())
            class_weights = {cls: total / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for _, label in dataset.imgs]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )