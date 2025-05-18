import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataLoaderBuilder:
    """
    Builds PyTorch DataLoaders for preprocessed image datasets.

    Args:
        data_root (Path): Root directory where preprocessed images are stored in class subfolders.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses used for data loading.
        shuffle (bool): Whether to shuffle the training data.
    """
    def __init__(self, data_root: Path, batch_size: int = 16, num_workers: int = 0, shuffle: bool = True):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def build_loader(self, subset: str):
        """
        Builds a DataLoader for a specific subset ('train', 'val', or 'test').

        Args:
            subset (str): Which subset to load ('train', 'val', 'test').

        Returns:
            DataLoader: PyTorch DataLoader for the specified subset.
        """
        subset_path = os.path.join(self.data_root, subset)
        dataset = datasets.ImageFolder(root=subset_path, transform=self.transform)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if subset == 'train' else False,
            num_workers=self.num_workers
        )