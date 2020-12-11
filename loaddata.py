import numpy as np
from torch.utils.data import *
from imutils import paths
import cv2

class DataLoader(Dataset):
    def __init__(self, img, imgSize, is_transform=None):
        self.img = img
        for i in range(len(img)):
            self.img_paths += [lpn for lpn in paths.list_images(img_dir[i])]
        self.img_size = imgShape
        self.is_transform = is_transfer

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (1,0,2))
        resizedImage /= 127.0
        return resizedImage, img_name
