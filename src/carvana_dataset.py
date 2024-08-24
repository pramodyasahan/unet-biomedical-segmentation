import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CarvanaDataset(Dataset):
    def __init__(self, root_dir, test=False):
        self.root_dir = root_dir

        if test:
            self.images = sorted([root_dir + "/manual_test/" + i for i in os.listdir(root_dir + "/manual_test/")])
            self.images = sorted(
                [root_dir + "/manual_test_mask/" + i for i in os.listdir(root_dir + "/manual_test_mask/")])
        else:
            self.images = sorted([root_dir + "train/" + i for i in os.listdir(root_dir + "train/")])
            self.images = sorted([root_dir + "train_mask/" + i for i in os.listdir(root_dir + "train_mask/")])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.images[index]).convert('L')

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
