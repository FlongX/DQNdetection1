import sys
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import create_parser
from agent import Agent
from Dataset.pancreas import NihDataset
from Dataset.mds import MDSDataset


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

    dataset_name = "MDS"

    dataset_config = {
        'Pancreas': {'dataset': NihDataset, 'root_path': r'./Data/NIH/', 'num_classes': 2,
                     'img_size': 224},
        'MDS': {'dataset': MDSDataset, 'root_path': r'./Data/MDS/', 'num_classes': 2,
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

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    device = 0
    agent = Agent(args, snapshot_path, device)

    agent.train()
