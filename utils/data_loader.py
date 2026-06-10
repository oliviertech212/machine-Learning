import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Class names must match your folder names exactly
CLASS_NAMES = ["bag", "food", "laptop"]

def get_transforms():
    """
    Training transforms - includes augmentation to make model stronger
    Testing transforms  - only resize and normalize
    """
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),      # randomly flip image
        transforms.RandomRotation(10),          # randomly rotate slightly
        transforms.ColorJitter(brightness=0.2), # vary brightness
        transforms.ToTensor(),                  # convert to numbers 0-1
        transforms.Normalize(                   # normalize pixel values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def get_dataloaders(dataset_path="dataset/", batch_size=32):
    """
    Loads images from dataset folder, splits into train/test,
    and returns DataLoaders ready for training.
    """
    train_transform, test_transform = get_transforms()

    # Load full dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)

    # Print what classes were found
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total images : {len(full_dataset)}")

    # Split 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, test_size])

    # Apply test transform to test data
    test_data.dataset.transform = test_transform

    print(f"Training images : {train_size}")
    print(f"Testing  images : {test_size}")

    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset.classes
