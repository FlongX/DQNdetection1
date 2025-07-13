import os
import cv2 as cv
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        z, x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class nih_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.image_list = []
        self.label_list = []

        image_dir = os.path.join(base_dir, 'image')
        label_dir = os.path.join(base_dir, 'label')

        case_dir = os.listdir(image_dir)
        # case_label_dir = os.listdir(label_dir)

        for case in case_dir:

            case_image_path = os.path.join(image_dir, case)
            case_label_path = os.path.join(label_dir, case)
            for num in os.listdir(case_image_path):
                self.image_list.append(os.path.join(case_image_path, num))
                self.label_list.append(os.path.join(case_label_path, num))

        print(self.image_list[0])
        print('image 长度：', len(self.image_list), 'label 长度： ', len(self.label_list))
        print(self.label_list[0])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_data_path = os.path.join(self.image_list[idx])
        label_data_path = os.path.join(self.label_list[idx])

        label_data = cv.imread(label_data_path)
        label_data = label_data[:, :, 0]
        label_data[label_data < 125] = 0
        label_data[label_data > 0] = 1

        image_data = cv.imread(image_data_path)
        image_data = image_data.transpose([2, 0, 1])  # c, h, w
        image_data = image_data / 255

        image, label = image_data, label_data

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample