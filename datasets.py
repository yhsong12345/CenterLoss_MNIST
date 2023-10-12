import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

# data constants
VALID_SPLIT = 1/6
NUM_WORKERS = 0



def create_datasets():
    """
    Function to build the training, validation, and testing dataset.
    """
    # transforms and augmentations for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # transforms for validation and testing
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # we choose the `train_dataset` and `valid_dataset` from the same...
    # ... distribution and later one divide 
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=train_transform)
    # this is the final test dataset to be used after training and validation completes
    dataset_test = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=valid_transform)
    # get the training dataset size, need this to calculate the...
    # number if validation images   
    train_dataset_size = len(train_dataset)
    # validation size = number of validation images
    valid_size = int(VALID_SPLIT*train_dataset_size)
    # all the indices from the training set
    indices = torch.randperm(len(train_dataset)).tolist()
    # final train dataset discarding the indices belonging to `valid_size` and after
    dataset_train = Subset(train_dataset, indices[:-valid_size])
    # final valid dataset from indices belonging to `valid_size` and after
    dataset_valid = Subset(train_dataset, indices[-valid_size:])
    print(f"Total training images: {len(dataset_train)}")
    print(f"Total validation images: {len(dataset_valid)}")
    print(f"Total test images: {len(dataset_test)}")
    return dataset_train, dataset_valid, dataset_test



def create_data_loaders(dataset_train, dataset_valid, dataset_test, BATCH_SIZE):
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader, test_loader