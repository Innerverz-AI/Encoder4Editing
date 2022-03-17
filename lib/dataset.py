import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster):
        self.datasets = []
        self.N = []
 
        for dataset_root in dataset_root_list:
            imgpaths_in_root = glob.glob(f'{dataset_root}/*.*g')

            for root, dirs, _ in os.walk(dataset_root):
                for dir in dirs:
                    imgpaths_in_root += glob.glob(f'{root}/{dir}/*.*g')

            self.datasets.append(imgpaths_in_root)
            self.N.append(len(imgpaths_in_root))

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        Xs = Image.open(image_path).convert("RGB")
        image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
        return self.transforms(Xs)

    def __len__(self):
        return sum(self.N)


class FaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        Xs = Image.open(self.source_path_list[idx]).convert("RGB")
        return self.transforms(Xs)

    def __len__(self):
        return len(self.source_path_list)
