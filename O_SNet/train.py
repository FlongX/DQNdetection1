import os
import torch
import shutil
import random
import numpy as np
from trainer import trainer
from O_SNet.net.unet_model import UNet
from O_SNet.config import create_parser
import torch.backends.cudnn as cudnn
from Dataset.pancreas import NihDataset
from Dataset.MDS import MDSDataset
from O_SNet.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from O_SNet.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

'''参数'''
parer = create_parser()
args = parer.parse_args()

if __name__ == "__main__":
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
    #args.model = 'TransUnet_pretrain'
    args.model = 'UNet'
    loss_name = 'ce_dice'
    #args.bbox_scale = 1
    args.bbox_scale = 1.5
    device = 0
    dataset_config = {
        # 'NIH': {'dataset': NihDataset, 'root_path': r'./Data/NIH/', 'num_classes': 2,
        #         'img_size': [224, 224]}}

        'MDS': {'dataset': MDSDataset, 'root_path': r'./Data/MDS/', 'num_classes': 2,
                'img_size': [224, 224]}}

    if args.batch_size != 24:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[args.Dataset_name]['num_classes']
    args.root_path = dataset_config[args.Dataset_name]['root_path']
    args.dataset = dataset_config[args.Dataset_name]['dataset']
    args.img_size = dataset_config[args.Dataset_name]['img_size']

    args.exp = args.Dataset_name + '_' + str(args.img_size[0])
    snapshot_path = "./model/{}/{}/{}/{}/".format(args.model, args.bbox_scale, args.exp, loss_name)
    snapshot_path = snapshot_path + 'epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    if not os.path.exists("./model/{}/{}/{}/{}/".format(args.model, str(args.bbox_scale), args.exp, loss_name)):
        os.makedirs("./model/{}/{}/{}/{}/".format(args.model, str(args.bbox_scale), args.exp, loss_name), exist_ok=True)
    else:
        shutil.rmtree("./model/{}/{}/{}/{}/".format(args.model, str(args.bbox_scale), args.exp, loss_name))
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

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

        if args.model == 'TransUnet_pretrain':
            net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer(args, net, snapshot_path, loss_name, device)
