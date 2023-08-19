import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models import get_encoder_architecture, AT, weight_scheduler
from datasets import get_pretraining_dataset
from evaluation import knn_predict
import setproctitle
import warnings

proc_title = "Genius"
setproctitle.setproctitle(proc_title)
warnings.filterwarnings('ignore', category=FutureWarning)


# train for one epoch, we refer to the implementation from: https://github.com/leftthomas/SimCLR
def train(snet, tnet, data_loader, train_optimizer, epoch, args):
    criterionAT = AT(2)
    snet.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True, device=args.device), im_2.cuda(non_blocking=True, device=args.device)
        feature_1, out_1 = snet(im_1)
        feature_2, out_2 = snet(im_2)
        # [2*B, D]
        feature_3, out_3 = tnet(im_1)
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        conloss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # loss = net(im_1, im_2, args)
        cloneloss = -torch.sum(feature_3 * feature_1, dim=-1).mean()
        tnet_train_list = []
        for name, module in tnet.f.f._modules.items():
            if name == '0':
                tnet_train_list.append(module(im_1))
            else:
                tnet_train_list.append(module(tnet_train_list[int(name) - 1]))
        tnet_train_list[-1] = F.normalize(tnet_train_list[-1], dim=-1)

        snet_train_list = []
        for name, module in snet.f.f._modules.items():
            if name == '0':
                snet_train_list.append(module(im_1))
            else:
                snet_train_list.append(module(snet_train_list[int(name) - 1]))
        snet_train_list[-1] = F.normalize(snet_train_list[-1], dim=-1)

        at4_loss = criterionAT(snet_train_list[6], tnet_train_list[6].detach()) * args.opt4
        at3_loss = criterionAT(snet_train_list[5], tnet_train_list[5].detach()) * args.opt3
        at2_loss = criterionAT(snet_train_list[4], tnet_train_list[4].detach()) * args.opt2
        at1_loss = criterionAT(snet_train_list[3], tnet_train_list[3].detach()) * args.opt1

        loss = conloss + at1_loss + at2_loss + at3_loss + at4_loss + cloneloss * args.opt5
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'
                .format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                        total_loss / total_num))

    return total_loss / total_num


# we use a knn monitor to check the performance of the pre-trained image encoder by following the implementation:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def test(net, memory_data_loader, test_data_clean_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


def get_dataloader():
    train_data, memory_data, test_data_clean = get_pretraining_dataset(args)
    _train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        drop_last=True
    )
    _memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True
    )
    _test_loader_clean = DataLoader(
        test_data_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True
    )
    return _train_loader, _memory_loader, _test_loader_clean


def parse():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH',
                        help='path to save the results (default: none)')
    parser.add_argument('--opt1', default=1000, type=int, help='opt1')
    parser.add_argument('--opt2', default=1000, type=int, help='opt2')
    parser.add_argument('--opt3', default=1000, type=int, help='opt3')
    parser.add_argument('--opt4', default=1000, type=int, help='opt4')
    parser.add_argument('--opt5', default=1, type=int, help='opt5')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--teacher', type=str, default='', metavar='PATH', help='bad teacher')
    parser.add_argument('--student', type=str, default='', metavar='PATH', help='good student')
    parser.add_argument('--ratio', type=float, default=0.04, help='the ratio of clean sample')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    # Set the random seeds and GPU information
    CUDA_LAUNCH_BLOCKING = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'data/{args.pretraining_dataset}/'
    print(args)
    # Dump args
    # Logging
    # results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    teacher = get_encoder_architecture(args).cuda(args.device)
    checkpoint = torch.load(args.teacher, map_location='cuda:0')
    if 'clip' in args.teacher:
        teacher.visual.load_state_dict(checkpoint['state_dict'])
    else:
        teacher.load_state_dict(checkpoint['state_dict'])

    student = get_encoder_architecture(args).cuda(args.device)
    # checkpoint = torch.load(args.student, map_location=f'cuda:{args.gpu}')
    # student.load_state_dict(checkpoint['state_dict'])

    # Define the optimizer
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = weight_scheduler(base_opt=[args.opt1, args.opt2, args.opt3, args.opt4],
                                 momentum_opt=10000, EPOCHS=200)
    train_loader, memory_loader, test_loader_clean = get_dataloader()
    epoch_start = 1

    # set params by mi
    mi = scheduler.estimate_mi(student, memory_loader, [3, 4, 5, 6], args.device)
    params = scheduler.update_weight(mi)
    args.opt1, args.opt2, args.opt3, args.opt4 = params[0], params[1], params[2], params[3]
    print('Estimated weight: ', params[0], params[1], params[2], params[3])

    # Training loop
    for epoch in range(epoch_start, args.epochs + 1):
        print("=================================================")
        train_loss = train(student, teacher, train_loader, optimizer, epoch, args)
        # if epoch % 10000000000000000 == 0:
            # mi = scheduler.estimate_mi(student, memory_loader, [3, 4, 5, 6], args.device)
            # params = scheduler.update_weight(mi)
            # args.opt1, args.opt2, args.opt3, args.opt4 = params[0], params[1], params[2], params[3]
        if epoch % 1000 == 0:
            torch.save({'epoch': epoch, 'state_dict': student.state_dict(), 'optimizer': optimizer.state_dict(), },
                       args.results_dir + '/model' + str(epoch) + '.pth')
