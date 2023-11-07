import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR_10_balanced(Dataset):
    def __init__(
        self, root="./data", is_train=True, transform=None, device=torch.device("cpu")
    ):
        dset_tmp = datasets.CIFAR10(
            root=root,
            train=is_train,
            transform=transform,
            download=True,
        )

        loader_tmp = DataLoader(
            dset_tmp, batch_size=len(dset_tmp), shuffle=True, num_workers=4
        )

        self.image, self.label = next(iter(loader_tmp))
        self.image, self.label = self.image.to(device), self.label.to(device)

    def __getitem__(self, index):
        return self.image[index], self.label[index]

    def __len__(self):
        return len(self.image)


def get_cifar_10_dataset(args, is_train=True, transform=None):
    if transform is None:
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
            ]
        )
    dset = CIFAR_10_balanced(
        root=args.data_dir,
        transform=transform,
        is_train=is_train,
        device=torch.device(args.device),
    )
    return dset


def select_train_valid_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_cifar_10_dataset(args, is_train=True)
    # use args.train_valid_ratio as train set, val set
    split_train_size = int((1 - args.train_valid_ratio) * (len(train_dataset)))
    split_valid_size = len(train_dataset) - split_train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [split_train_size, split_valid_size]
    )
    print(
        f"Train set size: {len(train_dataset)}, "
        f"validation set size: {len(val_dataset)}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_n_worker,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_n_worker,
        pin_memory=False,
    )
    return train_loader, val_loader

def select_test_loader(args):
    test_dataset = get_cifar_10_dataset(args, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_n_worker,
        pin_memory=False,
    )
    return test_loader
