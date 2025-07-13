import os
import torch
import random
import logging
import shutil
import numpy as np
from tqdm import tqdm
from utils import get_logger
import torch.multiprocessing
from config import create_parser
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from EvaluationMetrics.EM import local_test
from DataAugmentation.data_aug import RandomGenerator
torch.multiprocessing.set_sharing_strategy('file_system')
from net.unet_model import UNet
from Dataset.pancreas import NihDataset
from Dataset.MDS import MDSDataset
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def inference(args, model, test_log_path, device):
    db_test = args.dataset(base_dir=args.root_path, split="test",model_name=args.model, outsize=args.img_size, transform=transforms.Compose(
        [RandomGenerator(output_size=args.img_size,  split="test", bbox_scale=args.bbox_scale)]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    recall_sum = 0
    dice_sum = 0
    precision_sum = 0
    iou_sum = 0
    hd_sum = 0

    model.eval()

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_batch, label_batch, box = sampled_batch['image_l'], sampled_batch['label'], sampled_batch["box"]
        recall, dice, precision, iou, hd95, predict = local_test(image_batch, label_batch, box, args.img_size, model, device)

        image_batch = image_batch[0, 0:1, :, :]
        image_batch = (image_batch - image_batch.min()) / (image_batch.max() - image_batch.min())

        pred = torch.from_numpy(predict)
        pred = pred.unsqueeze(0)

        labs = label_batch * 50
        save_image(image_batch, test_log_path + 'image/' + str(i_batch) + '.png')
        save_image(pred.float(),
                   test_log_path + 'pred/' + str(i_batch) + '.png')
        save_image(labs.float(),
                   test_log_path + 'gt/' + str(i_batch) + '.png')
        logging.info('idx %d  recall %f dice %f precision %f iou %f' % (i_batch, recall, dice, precision, iou))

        recall_sum += recall
        dice_sum += dice
        precision_sum += precision
        iou_sum += iou
        hd_sum += hd95

    recall = recall_sum / len(db_test)
    dice = dice_sum / len(db_test)
    precision = precision_sum / len(db_test)
    iou = iou_sum / len(db_test)
    hd = hd_sum / len(db_test)

    logging.info('mean_recall %f mean_dice %f mean_precision %f mean_iou %f hd %f' % (recall, dice, precision, iou, hd))


if __name__ == "__main__":

    parer = create_parser()
    args = parer.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.Dataset_name = 'MDS'
    #args.model = 'TransUnet'
    # args.model = 'TransUnet_pretrain'
    args.model = 'UNet'
    #loss_name = 'Balance'
    loss_name = 'ce_dice'
    #args.bbox_scale = 1
    args.bbox_scale = 1.5
    device = 0
    dataset_config = {
        # 'Pancreas': {'dataset': NihDataset, 'root_path': r'/home/more/xfl/Data/Pancreas/', 'num_classes': 2, 'img_size': [224, 224]}

        'NIH': {'dataset': NihDataset, 'root_path': r'./Data/NIH/', 'num_classes': 2, 'img_size': [224, 224]},
        'MDS': {'dataset': MDSDataset, 'root_path': r'./Data/MDS/', 'num_classes': 2, 'img_size': [224, 224]}

    }
    if args.batch_size != 24:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[args.Dataset_name]['num_classes']
    args.root_path = dataset_config[args.Dataset_name]['root_path']
    args.dataset = dataset_config[args.Dataset_name]['dataset']
    args.img_size = dataset_config[args.Dataset_name]['img_size']

    args.exp = args.Dataset_name + '_' + str(args.img_size[0])
    snapshot_path = "./model/{}/{}/{}/{}/".format(args.model, str(args.bbox_scale), args.exp, loss_name)
    snapshot_path = snapshot_path + 'epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    test_log_path = "./test_log/{}/{}/{}/{}/".format(args.model, str(args.bbox_scale), args.exp, loss_name)
    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path, exist_ok=True)
        os.makedirs(test_log_path + 'image/', exist_ok=True)
        os.makedirs(test_log_path + 'pred/', exist_ok=True)
        os.makedirs(test_log_path + 'gt/', exist_ok=True)
    else:
        shutil.rmtree(test_log_path)
        os.makedirs(test_log_path, exist_ok=True)
        os.makedirs(test_log_path + 'image/', exist_ok=True)
        os.makedirs(test_log_path + 'pred/', exist_ok=True)
        os.makedirs(test_log_path + 'gt/', exist_ok=True)

    logger = get_logger(test_log_path + 'log.txt')
    logger.info(str(args))

    if args.model == 'UNet':
        net = UNet(n_channels=3, n_classes=args.num_classes).cuda(device=device)

    else:
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size[0] / args.vit_patches_size), int(args.img_size[0] / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda(device=device)

    net.load_state_dict(torch.load(snapshot_path + '/best.pth', map_location=torch.device(device=device)))

    inference(args, net, test_log_path, device)
