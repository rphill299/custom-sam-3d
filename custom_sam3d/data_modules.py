from monai.data import DataLoader
from monai.apps import DecathlonDataset

class DecathlonDataModule:
    """
    Simple reusable class to handle Decathlon datasets with MONAI.
    """

    def __init__(self,
                 root_dir,
                 task,
                 train_transform,
                 val_transform,
                 test_transform,
                 post_transform,
                 batch_size=1,
                 cache_rate=0.0,
                 num_workers=2,
                 download=True):

        self.root_dir = root_dir
        self.task = task
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.post_transform = post_transform
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.download = download

        # Build datasets
        self.train_ds = DecathlonDataset(
            root_dir=self.root_dir,
            task=self.task,
            transform=self.train_transform,
            section="training",
            download=self.download,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.val_ds = DecathlonDataset(
            root_dir=self.root_dir,
            task=self.task,
            transform=self.val_transform,
            section="validation",
            download=False,  # Only download once on training
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.test_ds = DecathlonDataset(
            root_dir=self.root_dir,
            task=self.task,
            transform=self.test_transform,
            section="validation",
            download=False,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        # Build loaders
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_train_ds(self):
        return self.train_ds

    def get_val_ds(self):
        return self.val_ds

    def get_test_ds(self):
        return self.test_ds

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader

    def _post_transform(self, data):
        return self.post_transform(data)
