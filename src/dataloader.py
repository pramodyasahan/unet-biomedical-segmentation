import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Dataloader(Dataset):
    def __init__(self, root_dir, test=False):
        self.root_dir = root_dir
        train_dir = os.path.join(root_dir, 'Training_Data/')
        test_dir = os.path.join(root_dir, 'Test_Data/')
        train_mask_dir = os.path.join(root_dir, 'Training_GroundTruth/')
        test_mask_dir = os.path.join(root_dir, 'Test_GroundTruth/')

        if test:
            print(f"Data is loading from {test_dir}")
            self.images = sorted([test_dir + i for i in os.listdir(root_dir + test_dir)])
            print(f"Data loaded")
            print(f"Data is loading from {test_mask_dir}")
            self.images = sorted(
                [test_mask_dir + i for i in os.listdir(test_mask_dir)])
            print(f"Data loaded")
        else:
            print(f"Data is loading from {train_dir}")
            self.images = sorted([train_dir + i for i in os.listdir(train_dir)])
            print(f"Data loaded")
            print(f"Data is loading from {train_mask_dir}")
            self.images = sorted([train_mask_dir + i for i in os.listdir(train_mask_dir)])
            print(f"Data loaded")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.images[index]).convert('L')

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
