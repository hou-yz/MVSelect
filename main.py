import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from multiview_detector.datasets import *
from multiview_detector.models.mvdet import MVDet
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # camera select module
    if args.select:
        args.alpha = 0
        # args.base_lr_ratio = 0.0
        # args.other_lr_ratio = 0.1
        args.lr = 1e-4 if args.lr is None else args.lr
    else:
        args.lr = 5e-4 if args.lr is None else args.lr

    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size, semi_supervised=args.semi_supervised,
                             dropout=args.dropcam, augmentation=args.augmentation)
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    if args.resume is None:
        logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}' \
                 f'{args.arch}_{"SL_" if args.select else ""}max_single_dropcam{args.dropcam}_lr{args.lr}_' \
                 f'base{args.base_lr_ratio}select{args.select_lr_ratio}other{args.other_lr_ratio}_' \
                 f'b{args.batch_size}_e{args.epochs}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        logdir = f'logs/{args.dataset}/{args.resume}'
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDet(train_set, args.arch, use_bottleneck=args.use_bottleneck, outfeat_dim=args.outfeat_dim).cuda()

    # load checkpoint
    if args.select:
        with open(f'logs/{args.dataset}/performance.txt', 'r') as fp:
            result_str = fp.read()
        print(result_str)
        load_dir = result_str.split('\n')[1]
        pretrained_dict = torch.load(f'{load_dir}/MultiviewDetector.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'cam_' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'cam_' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'cam_' in n and p.requires_grad],
                    "lr": args.lr * args.select_lr_ratio, }, ]
    # param_dicts = [{"params": [p for n, p in model.named_parameters() if 'cam_pred' in n and p.requires_grad], }]
    # optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=3):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    # if args.select:
    #     # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    #     # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
    #     #                                                 epochs=args.epochs)
    #     scheduler = None
    # else:
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
    #                                                     epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=args.lr*0.01,max_lr=args.lr,step_size_up=100,cycle_momentum=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args)

    def test_with_select(override=None):
        t0 = time.time()
        test_losses, modas = [], []
        for init_cam in np.arange(train_set.num_cam) if args.select or override is not None else [None]:
            print(f'init camera {init_cam}:')
            init_cam = torch.tensor([init_cam]).cuda() if init_cam is not None else None
            test_loss, moda = trainer.test(None, test_loader, res_fpath, init_cam, override)
            test_losses.append(test_loss)
            modas.append(moda)
        test_loss, moda = np.average(test_losses), np.average(modas)
        print(f'average moda: {moda:.2f}%, time: {time.time() - t0:.2f}')
        if override is not None:
            return modas
        return test_loss, moda

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []

    # learn
    res_fpath = os.path.join(logdir, 'test.txt')
    # trainer.test(0, test_loader, res_fpath)
    # test_with_select()

    if args.resume is None:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer.train(epoch, train_loader, optimizer, scheduler)
            print('Testing...')
            test_loss, moda = test_with_select()

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            test_loss_s.append(test_loss)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        model.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
        model.eval()
    print('Test loaded model...')
    if not args.select:
        modas = []
        for cam in range(train_set.num_cam):
            modas.append(test_with_select(override=cam))
        modas = np.array(modas)
        best2cam, avg2cam = np.mean(np.max(modas, axis=1)), np.mean(modas)
        print(' '.join(f'cam {np.argmax(modas[cam])} {np.max(modas[cam]):.2f} |' for cam in range(train_set.num_cam)))
        print(f'best {best2cam:.2f}, average {avg2cam:.2f}')
        _, moda = trainer.test(args.epochs, test_loader, res_fpath)
        np.savetxt(f'{logdir}/modas_{moda:.2f}_best{best2cam:.2f}_avg{avg2cam:.2f}.txt', modas, '%.2f',
                   header=f'loading checkpoint...\n'
                          f'{logdir}\n',
                   footer=' '.join(f'cam {np.argmax(modas[cam])} {np.max(modas[cam]):.2f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'2 best cam: {best2cam:.2f}, 2 average cam: {avg2cam:.2f}\n'
                          f'all cam: {moda:.2f}')
        if args.resume is None:
            shutil.copyfile(f'{logdir}/modas_{moda:.2f}_best{best2cam:.2f}_avg{avg2cam:.2f}.txt',
                            f'logs/{args.dataset}/performance.txt')

    else:
        trainer.test(args.epochs, test_loader, res_fpath)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--semi_supervised', type=float, default=0)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    # parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--select_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--select', type=str2bool, default=False)

    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
