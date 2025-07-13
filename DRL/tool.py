from DRL.config import Transition
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
import random
import matplotlib.patches as patches
import torch
from scipy.ndimage.interpolation import zoom
import pandas as pd
import numpy as np
from scipy import ndimage
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import UnivariateSpline
import torchvision.transforms as transforms

def IOU(box, gt):
    w1, w2, h1, h2 = box
    box_area = (w2 - w1) * (h2 - h1)

    inter_area = (gt[h1:h2, w1:w2] == 1).sum()
    union_area = box_area + (gt == 1).sum() - inter_area

    iou = inter_area / union_area
    return iou


def recall(box, gt):
    w1, w2, h1, h2 = box
    inter_area = (gt[h1:h2, w1:w2] == 1).sum()

    r = (inter_area + 0.000001) / ((gt == 1).sum() + 0.000001)
    return r


def calculate_position_box(actions, xmin=0, xmax=224, ymin=0, ymax=224):
    """
        执行所有操作， 生成最终标注框
    """

    def rewrap(coord):  # 防止超出边界
        return int(min(max(coord, 0), 224))

    for r in actions:
        if r == 1:  # zoom up_left
            xmax = xmax - (xmax - xmin) * 1/4
            ymax = ymax - (ymax - ymin) * 1/4

        if r == 2:  # zoom up_right
            xmin = xmin + (xmax - xmin) * 1/4
            ymax = ymax - (ymax - ymin) * 1/4

        if r == 3:  # zoom down_left
            xmax = xmax - (xmax - xmin) * 1/4
            ymin = ymin + (ymax - ymin) * 1/4

        if r == 4:  # zoom down_right
            xmin = xmin + (xmax - xmin) * 1/4
            ymin = ymin + (ymax - ymin) * 1/4

        if r == 5:  # zoom center
            w = xmax - xmin
            h = ymax - ymin
            xmin = xmin + w * 1/8
            xmax = xmax - w * 1/8
            ymin = ymin + h * 1/8
            ymax = ymax - h * 1/8

        if r == 6:  # shift left
            w = xmax - xmin
            xmin = xmin - w * 1/4
            xmax = xmax - w * 1/4
        if r == 7:  # shift right
            w = xmax - xmin
            xmin = xmin + w * 1/4
            xmax = xmax + w * 1/4

        if r == 8:  # shift up
            h = ymax - ymin
            ymin = ymin - h * 1/4
            ymax = ymax - h * 1/4

        if r == 9:  # shift down
            h = ymax - ymin
            ymin = ymin + h * 1/4
            ymax = ymax + h * 1/4

        xmin, xmax, ymin, ymax = rewrap(xmin), rewrap(xmax), rewrap(ymin), rewrap(ymax)
    return [xmin, xmax, ymin, ymax]


def show_new_bdbox(image, labels, color='r', count=0):
    """
        查看标注框对应的图片
    """
    # toPIL = transforms.ToPILImage()
    # image_data = toPIL(image)
    # plt.imshow(image_data)
    # plt.show()
    xmin, xmax, ymin, ymax = labels[0], labels[1], labels[2], labels[3]
    fig, ax = plt.subplots(1)
    image[image < 0] = 0.0
    image[image > 1] = 1.0
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration " + str(count))
    plt.savefig(str(count) + '.png', dpi=100)
    plt.close()

def eval_stats_at_threshold(pre_box, gt):
    w1, w2, h1, h2 = pre_box[0], pre_box[1], pre_box[2], pre_box[3]
    box_area = (w2 - w1) * (h2 - h1)

    inter_area = (gt[h1:h2, w1:w2] == 1).sum()
    union_area = box_area + (gt == 1).sum() - inter_area
    re = inter_area / ((gt == 1).sum())
    rm = inter_area / union_area

    return re, rm



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis + 1).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, axes=(1, 2), order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def elastic_transform(image, label, alpha, sigma,
                      alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)
    return imageC, labelC


class RandomGenerator(object):
    def __init__(self, output_size, split=None):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.split == 'train':

            if random.random() < 0.25:
                image, label = random_rot_flip(image, label)

            if random.random() < 0.25:
                image, label = random_rotate(image, label)
            # if random.random() < 0.5:
            #     image, label = elastic_transform(image[0, :, :], label, image.shape[1] * 2,
            #                                      image.shape[1] * 0.08,
            #                                      image.shape[1] * 0.08)

                # image = np.expand_dims(image, axis=0)
                # image = np.concatenate((image, image, image), axis=0)

        z, x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image).float() / 255.0  # ensure float and normalized
        # image = self.normalize(image)  # ImageNet style normalize
        label = torch.from_numpy(label)
        image = torch.from_numpy(image)
        sample = {'image': image, 'label': label.long()}
        return sample


def list2csv(list_, column_name, index_name, csv_save_path):
    column = [column_name]  # 列表头名称
    data = pd.DataFrame(columns=column, data=list_)  # 将数据放进表格
    data.index.name = index_name
    data.to_csv(csv_save_path)  # 数据存入csv,存储位置及文件名称


def show_reward_curve(csv_path, png_save_path, xlab, ylab, Smoothing):
    df = pd.read_csv(csv_path)

    # 设置图片大小
    plt.figure(figsize=(20, 12))

    # 设置背景网格
    plt.grid(linestyle='-.', linewidth=1.5, zorder=0)  # zorder控制绘图顺序，值越大绘图越晚

    # 设置x, y轴标签
    plt.xlabel(xlab, fontproperties='Times New Roman', fontweight='bold', fontsize=35, labelpad=15)
    plt.ylabel(ylab, fontweight='bold', fontproperties='Times New Roman', fontsize=35, labelpad=15)

    # 设置坐标轴刻度值

    plt.xticks(fontproperties='Times New Roman', size=30)
    plt.yticks(fontproperties='Times New Roman', size=30)

    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x / 10))))

    # 局部多项式拟合
    spline = UnivariateSpline(df.iloc[:, 0], df.iloc[:, 1], s=Smoothing)  # 调整参数 s 控制平滑度
    fit_epochs = np.linspace(1, len(df.iloc[:, 0]), 2 * len(df.iloc[:, 0]))  # 更密集的点来绘制曲线
    fit_loss = spline(fit_epochs)

    # 绘制
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], linestyle='dashdot', linewidth=2, marker='o', ms=12, alpha=0.3)
    plt.plot(fit_epochs, fit_loss, linestyle='-', linewidth=6, ms=12, color='#fa5050')

    # 保存和展示
    plt.savefig(png_save_path, dpi=300)
    plt.close()


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
