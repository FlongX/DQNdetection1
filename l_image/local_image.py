from DRL.config import create_parser as agent_create_parser
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from DRL.agent import Agent
import os
import cv2 as cv
from scipy.ndimage import zoom
import pandas as pd

device = 0
save_model_path = r'DQN1/DRL/model/DRL_MDS224/DRL_epo25_bs100_lr1e-06_s1234/policy24.pth'
parer1 = agent_create_parser()
args1 = parer1.parse_args()
agent = Agent(args1, save_model_path, device, './')
open('MDSbox_data.csv', 'w').close()
#
# base_dir = r'./Data/NIH/'
base_dir = r'./Data/MDS/'
image_dir = os.path.join(base_dir, 'image')
label_dir = os.path.join(base_dir, 'label')

case_dir = os.listdir(image_dir)
# case_label_dir = os.listdir(label_dir)

for case in case_dir:

    case_image_path = os.path.join(image_dir, case)
    case_label_path = os.path.join(label_dir, case)
    box_list = []
    num_list = []
    for num in os.listdir(case_image_path):
        image_data_path = os.path.join(case_image_path, num)
        image_data = cv.imread(image_data_path)
        image_data = image_data.transpose([2, 0, 1])  # c, h, w

        image = image_data
        z, y, x = image.shape
        image = zoom(image, (1, 224 / y, 224 / x), order=3)  # why not 3?
        o_image = image.copy()
        image = torch.from_numpy(image).cuda(device)
        box = agent.local_box(image)
        box_list.append(box)
        num_list.append(num)

        print(case, '- ', box, num)

    # 创建一个包含类别和序列信息的列索引
    row_index = [f'{case}_{seq}' for seq in num_list]

    df = pd.DataFrame(box_list, columns=["xmin", "xmax", "ymin", "ymax"], index=row_index)
    with open('MDSbox_data.csv', 'a') as f:
        df.to_csv(f)







