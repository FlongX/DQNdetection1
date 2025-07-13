import os
import cv2 as cv
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
import pandas as pd


class MDSDataset(Dataset):
    def __init__(self, base_dir, split, model_name, outsize, test_case=None, transform=None):
        self.transform = transform
        self.split = split
        self.image_list = []
        self.label_list = []
        self.box_list = []
        self.outsize = outsize

        image_dir = os.path.join(base_dir, 'image')
        label_dir = os.path.join(base_dir, 'label')
        box_dir = '../l_image/' + 'MDSbox_data.csv'
        df = pd.read_csv(box_dir, index_col=0)  # 假设CSV文件名为box_data.csv，第一列作为行索引

        case_dir = os.listdir(image_dir)
        if self.split == 'train':
            for case in case_dir:
                if int(case) <= 197:
                    case_image_path = os.path.join(image_dir, case)
                    case_label_path = os.path.join(label_dir, case)
                    for num in os.listdir(case_image_path):
                        # 构建列索引
                        column_name = f'{case}_{num}'
                        box = [int(df.loc[column_name].xmin), int(df.loc[column_name].xmax),
                               int(df.loc[column_name].ymin), int(df.loc[column_name].ymax)]
                        if box[1] - box[0] == 0 or box[3] - box[2] == 0:  # 排除粗分割的预测结果为0的情况
                            continue
                        label_data = cv.imread(os.path.join(case_label_path, num))
                        label_data = label_data[:, :, 0]
                        label_data[label_data < 125] = 0
                        label_data[label_data > 0] = 1
                        x, y = label_data.shape
                        label_data = zoom(label_data, (self.outsize[0] / x, self.outsize[1] / y), order=0)
                        if label_data[box[2]:box[3], box[0]:box[1]].sum() == 0:
                            continue
                        self.image_list.append(os.path.join(case_image_path, num))
                        self.label_list.append(os.path.join(case_label_path, num))
                        self.box_list.append(box)
        if self.split == 'val':
            for case in case_dir:
                if 197 < int(case) <= 225:
                    case_image_path = os.path.join(image_dir, case)
                    case_label_path = os.path.join(label_dir, case)
                    for num in os.listdir(case_image_path):

                        self.image_list.append(os.path.join(case_image_path, num))
                        self.label_list.append(os.path.join(case_label_path, num))
                        # 构建列索引

                        column_name = f'{case}_{num}'

                        box = [int(df.loc[column_name].xmin), int(df.loc[column_name].xmax),
                               int(df.loc[column_name].ymin), int(df.loc[column_name].ymax)]
                        self.box_list.append(box)
        if self.split == 'test':
            for case in case_dir:
                if 225 < int(case):
                    case_image_path = os.path.join(image_dir, case)
                    case_label_path = os.path.join(label_dir, case)
                    for num in os.listdir(case_image_path):

                        self.image_list.append(os.path.join(case_image_path, num))
                        self.label_list.append(os.path.join(case_label_path, num))
                        # 构建列索引

                        column_name = f'{case}_{num}'

                        box = [int(df.loc[column_name].xmin), int(df.loc[column_name].xmax),
                               int(df.loc[column_name].ymin), int(df.loc[column_name].ymax)]
                        self.box_list.append(box)

        print(self.image_list[0])
        print('image 长度：', len(self.image_list), 'label 长度： ', len(self.label_list))
        print(self.label_list[0])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_data_path = os.path.join(self.image_list[idx])
        label_data_path = os.path.join(self.label_list[idx])

        box_data = self.box_list[idx]
        label_data = cv.imread(label_data_path)
        label_data = label_data[:, :, 0]
        label_data[label_data < 125] = 0
        label_data[label_data > 0] = 1

        image_data = cv.imread(image_data_path)
        image_data = image_data.transpose([2, 0, 1])  # c, h, w

        image, label = image_data, label_data

        sample = {'image': image, 'label': label, 'box': box_data}
        if self.transform:
            sample = self.transform(sample)

        return sample
