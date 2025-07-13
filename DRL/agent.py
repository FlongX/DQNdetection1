import shutil
from DRL.net import *
from DRL.tool import *
from scipy.ndimage.interpolation import zoom
import os
import imageio
import random
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim as optim
from DRL.config import *
import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

criterion = nn.MSELoss().cuda(device=0)


class Agent(object):
    def __init__(self, args, save_model_path, device, test_log_path=None):
        """
            初始化参数和网络
        """
        self.args = args
        self.n_steps = args.n_steps
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.save_path = save_model_path
        self.n_actions = args.n_actions
        self.nu = args.nu
        self.T_Recall = args.T_Recall
        self.T_IOU = args.T_IOU

        self.eps = args.eps
        self.target_update = args.target_update

        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.feature_extractor = FeatureExtractor()

        self.target_net.eval()

        self.feature_extractor = self.feature_extractor.cuda(device=device)
        self.policy_net = self.policy_net.cuda(device=self.device)
        self.target_net = self.target_net.cuda(device=self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = ReplayMemory(args.memory)

        self.actions_history = torch.zeros((self.n_steps, self.n_actions))
        self.num_episodes = args.num_episodes

        self.test_log_path = test_log_path
        if self.test_log_path is None:
            self.logger = get_logger(self.save_path + "/log.txt")
        else:
            self.logger = get_logger(self.test_log_path + "/log.txt")

        self.idx = 0

    @staticmethod
    def compute_reward(actual_box, previous_box, gt):
        a = IOU(actual_box, gt)
        b = IOU(previous_box, gt)
        rm = a - b
        if rm <= 0:
            return -1, a, b
        return 1, a, b

    def compute_trigger_reward(self, actual_box, gt):
        rt = recall(actual_box, gt)
        rm = IOU(actual_box, gt)

        if rt >= self.T_Recall and rm >= self.T_IOU:
            return self.nu
        return -1 * self.nu

    def get_best_next_action(self, actions, gt):  # 探索阶段不适用Q_net选择动作，从积极动作集中选一个最好的
        positive_actions = []
        negative_actions = []
        actual_box = calculate_position_box(actions)
        for i in range(0, 10):
            copy_actions = actions.copy()
            copy_actions.append(i)  # 在历史动作集中添加一个动作([0,8])
            new_equivalent_coord = calculate_position_box(copy_actions)  # 经过copy_actions后的标注框坐标
            if i != 0:
                reward, _, __ = self.compute_reward(new_equivalent_coord, actual_box, gt)
            else:  # 动作0是停止标志
                reward = self.compute_trigger_reward(new_equivalent_coord, gt)

            # 根据reward判断动作的好坏并保存
            if reward >= 0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions) == 0:  # 没有好动作 -> 随机选一个动作
            return random.choice(negative_actions)
        return random.choice(positive_actions)  # 随机选一个好动作

    def select_action(self, state, actions, gt):
        sample = random.random()
        eps_t = self.eps

        if sample > eps_t:  # 1-eps的概率 根据策略网络选择最优动作
            with torch.no_grad():
                s = state
                q_v = self.policy_net(s)
                _, max_q = torch.max(q_v.data, 1)
                action = max_q[0]
                try:
                    return action.cpu().numpy()[0]
                except:
                    return action.cpu().numpy()
        else:
            return random.randint(0, self.n_actions - 1)
            #return self.get_best_next_action(actions, gt)

    def select_action_test(self, state):
        with torch.no_grad():
            s = state
            q_v = self.policy_net(s)
            _, q_max = torch.max(q_v.data, 1)
            action = q_max[0]
            return action

    def optimize_model(self, episodes, n_batch):
        if len(self.memory) < self.batch_size:
            return 0

        self.idx += 1
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 判断是否存在s(t+1)
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        # torch.Size([100]); tensor类型,100个值,是否为None; 将batch.next_state(100个)通过lambda表达式,list->tuple->tensor;
        next_states = [s for s in batch.next_state if s is not None]

        non_final_next_states = torch.cat(next_states).type(Tensor)

        # 获取（s(t), a, r, s(t + 1)）
        state_batch = torch.cat(batch.state).type(Tensor).cuda(device=self.device)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).type(LongTensor).cuda(device=self.device)
        reward_batch = torch.FloatTensor(batch.reward).view(-1, 1).type(Tensor).cuda(device=self.device)
        next_states_values = torch.zeros(self.batch_size, 1).type(Tensor).cuda(device=self.device)
        non_final_next_states = non_final_next_states.cuda(device=self.device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # 获取策略网络中和action_batch对应的q值
        with torch.no_grad():
            d = self.target_net(non_final_next_states)
            next_states_values[non_final_mask] = d.max(1)[0].view(-1, 1)
            expected_state_action_values = (next_states_values * self.gamma) + reward_batch

        loss = criterion(state_action_values, expected_state_action_values)

        if self.idx % 100 == 0:
            self.logger.info('episodes : %d n_batch : %d loss : %f' % (episodes, n_batch, loss.item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compose_state(self, image, dtype=FloatTensor):
        """
             状态组合 = 图片特征向量 + 历史动作集
        """
        image_feature = self.get_features(image, dtype)  # 获取从图片中提取得到的特征(维度[1, 512, 7, 7])
        image_feature = image_feature.view(1, -1)  # 修改维度为1行(维度[1, 25088])
        history_flatten = self.actions_history.view(1, -1).type(dtype)
        history_flatten = history_flatten.cuda(device=self.device)
        state = torch.cat((image_feature, history_flatten), 1)  # 按列拼接(左右拼接), shape=[1, 25169]
        return state

    def get_features(self, image, dtype=FloatTensor):
        global transform
        image = image.view(1, *image.shape)  # 使用*将image.shape中的数据直接读出(image.shape是使用张量存储的数据); 使用view重构,(1,3,224,224)
        image = image.type(dtype).cuda(device=self.device)
        feature = self.feature_extractor(image)
        return feature.data

    def update_history(self, action):
        """
            添加新的动作到历史动作集(共保存10个动作组)
            参数 :
                - 动作索引
        """
        size_history_vector = len(torch.nonzero(self.actions_history))  # 历史动作集中非0动作的个数
        if size_history_vector < self.n_steps:
            self.actions_history[size_history_vector][action] = 1  # 不足10个动作 -> 将动作添加到最后一个动作中(最后一个动作集中动作action设为1)

        return self.actions_history

    def local_box(self, image):

        self.policy_net.load_state_dict(torch.load(self.save_path, map_location=torch.device(device=self.device)))
        # Q-Network转换到评估模式
        self.policy_net.eval()

        xmin = 0
        xmax = self.args.img_size
        ymin = 0
        ymax = self.args.img_size

        done = False
        all_actions = []
        self.actions_history = torch.zeros((self.n_steps, self.n_actions))
        # 执行10步 或 选择动作0
        steps = 0

        original_image = image.clone()
        state = self.compose_state(image)

        while not done:
            steps += 1
            action = self.select_action_test(state)
            all_actions.append(action)  # 收集选择的动作
            if action == 0:
                next_state = None
                new_equivalent_coord = calculate_position_box(all_actions, xmin, xmax, ymin, ymax)
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = calculate_position_box(all_actions, xmin, xmax, ymin, ymax)

                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]),
                            int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break

                next_state = self.compose_state(new_image)

            if steps == self.n_steps:
                done = True

            state = next_state

        fin_xmin, fin_xmax, fin_ymin, fin_ymax = new_equivalent_coord
        box = [fin_xmin, fin_xmax+ 1, fin_ymin, fin_ymax+1]

        return box

    def predict_image(self, image, media_path, gt, plot=True):
        """
            预测图像的标注框
        """
        self.policy_net.load_state_dict(torch.load(self.save_path, map_location=torch.device(device=self.device)))
        # Q-Network转换到评估模式
        self.policy_net.eval()

        xmin = 0
        xmax = self.args.img_size
        ymin = 0
        ymax = self.args.img_size

        done = False
        all_actions = []
        self.actions_history = torch.zeros((self.n_steps, self.n_actions))
        # 执行10步 或 选择动作0
        steps = 0

        original_image = image.clone()
        state = self.compose_state(image)

        while not done:
            steps += 1
            action = self.select_action_test(state)
            all_actions.append(action)  # 收集选择的动作
            if action == 0:
                next_state = None
                new_equivalent_coord = calculate_position_box(all_actions, xmin, xmax, ymin, ymax)
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = calculate_position_box(all_actions, xmin, xmax, ymin, ymax)
                print(recall(new_equivalent_coord, gt), IOU(new_equivalent_coord, gt))
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]),
                            int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break

                next_state = self.compose_state(new_image)

            if steps == self.n_steps:
                done = True

            state = next_state

            if plot:
                show_image = original_image.clone()
                show_image[show_image < 0] = 0
                show_image[show_image > 255] = 255
                show_image = show_image / 255
                show_new_bdbox(show_image, new_equivalent_coord, color='b', count=steps)

        if plot:
            tested = 0
            while os.path.isfile(media_path + '/movie_' + str(tested) + '.gif'):
                tested += 1

            fp_out = media_path + '/movie_' + str(tested) + '.gif'
            images = []
            for count in range(1, steps + 1):
                images.append(imageio.imread(str(count) + '.png'))

            imageio.mimsave(fp_out, images)

            for count in range(1, steps):
                os.remove(str(count) + '.png')

        return new_equivalent_coord, steps

    def save_network(self, epoch):
        """
            保存Q-Network
        """
        save_mode_path = os.path.join(self.save_path, 'policy' + str(epoch) + '.pth')
        torch.save(self.policy_net.state_dict(), save_mode_path)
        self.logger.info('Saved')

    def load_network(self):
        """
            调用现有的Q网络
        """
        return torch.load(self.save_path)

    def train(self):
        """
            训练数据集
        """
        db_train = self.args.dataset(base_dir=self.args.root_path, split="train", outsize=self.args.img_size,
                                     transform=transforms.Compose(
                                         [RandomGenerator(output_size=[self.args.img_size, self.args.img_size],
                                                          split='train')]))

        trainloader = DataLoader(db_train, batch_size=self.args.train_batch_size, shuffle=True,
                                 num_workers=multiprocessing.cpu_count(),
                                 pin_memory=False)

        max_iterations = self.args.num_episodes * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
        len_set = len(db_train)

        logging.info(
            str(self.args) + '\nThe length of train set is: {}\n\n============================================\n'.format(
                len(db_train)))
        self.logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

        iterator = tqdm(range(self.num_episodes), ncols=70)
        # 初始的边框是整张图片
        xmin = 0
        xmax = self.args.img_size
        ymin = 0
        ymax = self.args.img_size

        n_batch = 0
        rewards = []
        loss = []
        loss_ = 0
        for i_episode in iterator:
            self.logger.info("Episode " + str(i_episode))
            for sampled_batch in trainloader:
                image_batch, label_batch = sampled_batch['image'][0, :], sampled_batch['label'][0, :]
                if label_batch.max() == 0:
                    continue
                else:
                    n_batch += 1
                    rew = 0
                    original_image = image_batch.clone()  # 保留一张原始图片; 完全拷贝, 保存image的所有信息到original_image

                    gt = label_batch  # 图片中的第一个标注框
                    all_actions = []

                    # 初始化环境和状态空间
                    self.actions_history = torch.zeros((self.n_steps, self.n_actions))
                    state = self.compose_state(image_batch)  # 状态 = 图片特征向量 + 动作历史
                    original_coordinates = [xmin, xmax, ymin, ymax]  # 初始的标注框
                    done = False  # 一回合结束标志
                    t = 0
                    actual_equivalent_coord = original_coordinates  # 实际标注框初始化
                    new_equivalent_coord = original_coordinates  # 新标注框初始化

                    while not done:  # 结束一张图片的标志：执行10次动作 或 action=0
                        t += 1
                        action = self.select_action(state, all_actions, gt)

                        all_actions.append(action)  # 收集选择的动作

                        if action == 0:  # 结束标志
                            next_state = None
                            new_equivalent_coord = calculate_position_box(all_actions)  # 执行所有的动作获得标注框坐标
                            reward = self.compute_trigger_reward(new_equivalent_coord,
                                                                 gt)  # 新标注框和最佳标注框的差距; 不低于阈值->3, 低于阈值->(-3)
                            done = True  # 一回合结束

                        else:

                            self.actions_history = self.update_history(action)  # 更新历史动作集

                            new_equivalent_coord = calculate_position_box(all_actions)  # 根据历史动作集计算标注框坐标

                            new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]),
                                        int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]  # 标注框框住的图片(先y后x)
                            # print("new image before transform" + str(new_image.shape))
                            try:
                                new_image = transform(new_image)
                            except ValueError:
                                break

                            next_state = self.compose_state(new_image)  # 状态 = 图片特征向量 + 动作历史
                            reward, a, b = self.compute_reward(new_equivalent_coord, actual_equivalent_coord,
                                                               gt)  # 新旧两个标注框和最佳标注框的差距 计算奖励

                            actual_equivalent_coord = new_equivalent_coord  # 更新标注框

                        # print(action, t, 'reward: ', reward, 'done:', done, 'new: ', new_equivalent_coord, 'old: ', actual_equivalent_coord, gt_box)
                        rew += reward

                        s1 = state.clone()
                        s1 = s1.cpu().detach()
                        s2 = None
                        if next_state is not None:
                            s2 = next_state.clone()
                            s2 = s2.cpu().detach()

                        self.memory.push(s1, int(action), s2, reward)  # 更新缓存

                        state = next_state
                        loss_ = self.optimize_model(i_episode, n_batch)

                        if t == self.n_steps:
                            done = True

                if n_batch % (int(len_set / 10)) == 0:
                    rewards.append(rew)
                    loss.append(loss_)

            if i_episode % self.target_update == 0:  # 训练完一轮图片,使用策略网络的参数更新目标网络
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode < 9:
                self.eps -= 0.1

            self.logger.info('eps: ' + str(self.eps) + '\n memory_len: ' + str(len(self.memory)))

            list2csv(rewards, column_name='reward', index_name='batch', csv_save_path=self.save_path + '/rewards.csv')
            list2csv(loss, column_name='loss', index_name='batch', csv_save_path=self.save_path + '/loss.csv')
            show_reward_curve(csv_path=self.save_path + '/rewards.csv',
                              png_save_path=self.save_path + '/rewards_curve.png', xlab='Epoch', ylab='Rewards',
                              Smoothing=100000)
            show_reward_curve(csv_path=self.save_path + '/loss.csv', png_save_path=self.save_path + '/loss_curve.png',
                              xlab='Epoch', ylab='Loss', Smoothing=10)

            if i_episode % 10 == 0 or i_episode == self.args.num_episodes - 1:
                self.save_network(i_episode)

                self.logger.info('\n!!!\n--updata_target_net--\n')
            self.logger.info('Complete')  # 完成一个epoch

    def test(self):
        media_path = 'media/' + self.args.Dataset_name + str(self.args.img_size)
        if not os.path.exists(media_path):
            os.makedirs(media_path, exist_ok=True)
        else:
            shutil.rmtree(media_path)
            os.makedirs(media_path, exist_ok=True)

        idx, case_idx = 0, 0
        recall_, iou_ = 0, 0
        max_recall, max_iou = 0, 0
        mean_recall, mean_iou = 0, 0
        min_recall, min_iou = 1, 1
        t_sum = 0

        self.logger.info("Predicting boxes...")

        test_image_dir = os.path.join(self.args.root_path, 'image')
        test_case_dir = os.listdir(test_image_dir)
        for test_case in test_case_dir:

            if 66 < int(test_case):
                os.makedirs(os.path.join(media_path, test_case), exist_ok=True)
                db_test = self.args.dataset(base_dir=self.args.root_path, split="test", outsize=self.args.img_size,
                                            transform=transforms.Compose(
                                                [RandomGenerator(
                                                    output_size=[self.args.img_size, self.args.img_size])]),
                                            test_case=test_case)

                testloader = DataLoader(db_test, batch_size=self.args.train_batch_size, shuffle=False,
                                        num_workers=8,
                                        pin_memory=True)

                self.logger.info('case: {}\n-----\n'.format(test_case))
                case_idx += 1

                test_case_recall = 0
                test_case_iou = 0
                test_case_id = 0

                for sampled_batch in testloader:
                    idx += 1
                    test_case_id += 1

                    image_batch, label_batch = sampled_batch['image'][0, :], sampled_batch['label'][0, :]

                    bbox, t = self.predict_image(image_batch, os.path.join(media_path, test_case), label_batch)  # 预测图像的标注框

                    re, rm = eval_stats_at_threshold(bbox, label_batch)
                    self.logger.info(
                        'idx: {} recall: {} IOU: {} t:{}'.format(idx - 1, re, rm, t))  # 根据阈值保存 平均精度AP和召回率recall
                    t_sum += t
                    recall_ += re
                    test_case_recall += re
                    iou_ += rm
                    test_case_iou += rm

                test_case_recall = test_case_recall / test_case_id
                test_case_iou = test_case_iou / test_case_id
                mean_recall += test_case_recall
                mean_iou += test_case_iou

                self.logger.info("mean recall: {} mean IOU: {}.".format(test_case_recall, test_case_iou))
                if min_recall > test_case_recall:
                    min_recall = test_case_recall
                if min_iou > test_case_iou:
                    min_iou = test_case_iou
                if max_recall < test_case_recall:
                    max_recall = test_case_recall
                if max_iou < test_case_iou:
                    max_iou = test_case_iou

        self.logger.info("Computing recall and ap...")
        recall_ = recall_ / idx
        iou_ = iou_ / idx
        self.logger.info("Final result : \n" + str(recall_) + str(iou_))  # 输出指标
        self.logger.info(
            "min_recall: {}  min_IOU: {} \n max_recall: {} maxIOU: {}\n mean_recall: {} mean_IOU: {} \n mean_t: {}".format(
                min_recall, min_iou, max_recall,
                max_iou, mean_recall / case_idx, mean_iou / case_idx, t_sum / idx))  # 输出指标
        return 'fin'
