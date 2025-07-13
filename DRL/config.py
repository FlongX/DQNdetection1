import torch
from collections import namedtuple
import torchvision.transforms as transforms
import argparse
import logging
import sys


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='root dir for data')
    parser.add_argument('--dataset', type=str, help='experiment_name')
    parser.add_argument('--num_episodes', type=int,
                        default=25, help='maximum episodes number to train')
    parser.add_argument('--batch_size', type=int,
                        default=100, help='batch_size per gpu')
    parser.add_argument('--target_update', type=int,
                        default=1, help='target update per target update')
    parser.add_argument('--eps', type=int,
                        default=1, help='Exploration probability')
    parser.add_argument('--train_batch_size', type=int,
                        default=1, help='batch_size per gpu of data')
    parser.add_argument('--T_Recall', type=int,
                        default=0.9, help='recall threshold')
    parser.add_argument('--T_IOU', type=int,
                        default=0.1, help='iou threshold')
    parser.add_argument('--nu', type=int,
                        default=3, help='trigger reward')
    parser.add_argument('--memory', type=int,
                        default=50000, help='replay memory')
    parser.add_argument('--gamma', type=float,
                        default=0.9, help='discount factor')
    parser.add_argument('--n_actions', type=int,
                        default=10, help='actions number')
    parser.add_argument('--n_steps', type=int,
                        default=10, help='actions number')

    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Q-network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')

    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    return parser


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%m%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
