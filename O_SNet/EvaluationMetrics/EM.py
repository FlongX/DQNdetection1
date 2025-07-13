import numpy as np
from medpy import metric
import torch
from scipy.ndimage import zoom


def Recall(predict, target):
    predict[predict > 0] = 1

    predict = np.atleast_1d(predict.astype(bool))
    target = np.atleast_1d(target.astype(bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def Precision(predict, target):
    predict[predict > 0] = 1

    predict = np.atleast_1d(predict.astype(bool))
    target = np.atleast_1d(target.astype(bool))

    tp = np.count_nonzero(predict & target)
    fp = np.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def Dice(predict, target):
    predict[predict > 0] = 1
    if predict.sum() > 0 and target.sum() > 0:
        dice = metric.binary.dc(predict, target)
        return dice
    else:
        return 0


def HD95(predict, target):
    predict[predict > 0] = 1
    target[target > 0] = 1
    if predict.sum() > 0 and target.sum() > 0:
        hd95 = metric.binary.hd95(predict, target)
        return hd95
    else:
        return 0


def IOU(predict, target):
    predict[predict > 0] = 1

    predict = np.atleast_1d(predict.astype(bool))
    target = np.atleast_1d(target.astype(bool))

    tp = np.count_nonzero(predict & target)
    fp = np.count_nonzero(predict & ~target)
    fn = np.count_nonzero(~predict & target)

    try:
        iou = tp / float(tp + fp + fn)
    except ZeroDivisionError:
        iou = 0.0
    return iou


def test_single_volume(image, target, net, device):
    target = target.squeeze(0)
    target = target.numpy()
    image = image.cuda(device=device)

    net.eval()
    with torch.no_grad():
        predict = net(image)
        predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).squeeze(0)
        predict = torch.tensor(predict)
        predict = predict.cpu().detach().numpy()

    recall = Recall(predict, target)
    dice = Dice(predict, target)
    precision = Precision(predict, target)
    iou = IOU(predict, target)
    hd95 = HD95(predict, target)
    return recall, dice, precision, iou, hd95, predict


def local_test(image, target, box, img_size, net, device):
    if box == [0, 0, 0, 0]:
        recall = 0
        dice = 0
        precision = 0
        iou = 0
        hd95 = 0
        predict = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        return recall, dice, precision, iou, hd95, predict

    target = target.squeeze(0)
    target = target.numpy()
    image = image.cuda(device=device)

    net.eval()
    with torch.no_grad():
        predict = net(image)
        predict = torch.argmax(torch.softmax(predict, dim=1), dim=1).squeeze(0)
        predict = predict.cpu().detach().numpy()

    l = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
    w = int(box[1]) - int(box[0])
    h = int(box[3]) - int(box[2])

    # 计算缩放因子
    zoom_factors = (h / img_size[0], w / img_size[1])
    # 用最近邻插值（order=0）保持类别不变

    resized_predict = zoom(predict, zoom=zoom_factors, order=0)
    l[box[2]: box[3], box[0]: box[1]] = resized_predict

    predict = l
    recall = Recall(predict, target)
    dice = Dice(predict, target)
    precision = Precision(predict, target)
    iou = IOU(predict, target)
    hd95 = HD95(predict, target)
    return recall, dice, precision, iou, hd95, predict