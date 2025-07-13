import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
from agent import Agent
from config import *
from Dataset.pancreas import NihDataset
from Dataset.mds import MDSDataset


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

    dataset_name = "NIH"

    dataset_config = {
        'NIH': {'dataset': NihDataset, 'root_path': r'/home/xfl/Data/NIH/', 'num_classes': 2,
                     'img_size': 224},
        'MDS': {'dataset': MDSDataset, 'root_path': r'/home/xfl/Data/MDS/', 'num_classes': 2,
                     'img_size': 224},
    }
    args.root_path = dataset_config[dataset_name]['root_path']
    args.dataset = dataset_config[dataset_name]['dataset']
    args.Dataset_name = dataset_name

    args.exp = 'DRL_' + dataset_name + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'DRL')
    snapshot_path = snapshot_path + '_epo' + str(args.num_episodes)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    log_folder = './test_log/' + args.exp
    os.makedirs(log_folder, exist_ok=True)

    model_path = snapshot_path + '/policy24.pth'
    device=0
    agent = Agent(args, model_path, device, log_folder)
    agent.test()
