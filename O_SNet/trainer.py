import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from DataAugmentation.data_aug import RandomGenerator
from O_SNet.utils import get_logger, EarlyStopping
from torch.nn.modules.loss import CrossEntropyLoss
from O_SNet.loss.loss import DiceLoss
from O_SNet.EvaluationMetrics.EM import local_test


def trainer(args, model, snapshot_path, loss_name, device):
    logger = get_logger(snapshot_path + "/log.txt")
    base_lr = args.base_lr
    lr_ = base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_epoch = args.max_epochs
    iter_num = 0

    iterator = tqdm(range(max_epoch), ncols=70)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ES = EarlyStopping(patience=20, patience_lr=10)

    writer = SummaryWriter(snapshot_path + '/log')

    db_train = args.dataset(base_dir=args.root_path, split="train", model_name=args.model, outsize=args.img_size,
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=args.img_size, split='train', bbox_scale=args.bbox_scale)]))
    db_val = args.dataset(base_dir=args.root_path, split="val", model_name=args.model, outsize=args.img_size,
                          transform=transforms.Compose(
                              [RandomGenerator(output_size=args.img_size, split='val', bbox_scale=args.bbox_scale)]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=8)

    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    logger.info(str(args) + '\nThe length of train set is: {}\n\n===========================\n'.format(len(db_train)))
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    for epoch_num in iterator:
        model.train()
        for sampled_batch in trainloader:
            image_batch, label_batch, f_n, b_n = sampled_batch['image_l'], sampled_batch['label_l'], \
                sampled_batch['f_n'], sampled_batch['b_n']
            image_batch, label_batch, f_n, b_n = image_batch.cuda(device=device), label_batch.cuda(
                device=device), f_n.cuda(device=device), b_n.cuda(device)
            seg_out = model(image_batch)

            loss_dice = dice_loss(seg_out, label_batch)
            loss_ce = ce_loss(seg_out, label_batch[:].long())
            loss = (loss_ce + loss_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1

            logger.info('iteration %d :  loss: %.4f' % (
                iter_num, loss.item()))

            if iter_num % 200 == 0:
                image = image_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image_g', image, iter_num)

                outputs = torch.argmax(torch.softmax(seg_out, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

                labs_g = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth_g', labs_g, iter_num)

        model.eval()
        dice_sum = 0
        for sampled_batch in valloader:
            image_batch, label_batch, box = sampled_batch['image_l'], sampled_batch['label'], sampled_batch["box"]
            _, dice, _, _, _, _ = local_test(image_batch, label_batch, box, args.img_size, model, device)
            dice_sum += dice

        val_dice = dice_sum / len(db_val)
        logger.info("val_dice {}".format(val_dice))

        decay, early_stop, save_model = ES.EStop(val_dice)

        if save_model:
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best' + '.pth'))
            logger.info("save model to {}".format(os.path.join(snapshot_path, 'best' + '.pth')))
        if early_stop:
            logger.info('EarlyStopping')
            break
        lr_ = lr_ * decay
        if lr_ < 1e-6:
            lr_ = 1e-6

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        logger.info(lr_)

    iterator.close()
    writer.close()
    return "Training Finished!"
