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
        # args.dropcam = 0.5
        args.base_lr_ratio = 0.1 if args.base_lr_ratio is None else args.base_lr_ratio
        args.other_lr_ratio = 0.1 if args.other_lr_ratio is None else args.other_lr_ratio
        args.lr = 1e-4 if args.lr is None else args.lr
    else:
        args.lr = 5e-4 if args.lr is None else args.lr
        args.base_lr_ratio = 1.0 if args.base_lr_ratio is None else args.base_lr_ratio
        args.other_lr_ratio = 1.0 if args.other_lr_ratio is None else args.other_lr_ratio

    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, split='train', world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size,
                             dropout=args.dropcam, augmentation=args.augmentation)
    val_set = frameDataset(base, split='train', world_reduce=args.world_reduce,
                           img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                           img_kernel_size=args.img_kernel_size)
    test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    if args.resume is None or not args.eval:
        select_settings = f'single_hard{int(args.hard)}_conf{args.beta_conf}even{args.beta_even}_'
        lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}' + \
                      f'select{args.select_lr_ratio}' if args.select else ''
        logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}{args.arch}_{args.aggregation}_' \
                 f'lr{args.lr}{lr_settings}_b{args.batch_size}_e{args.epochs}_dropcam{args.dropcam}_' \
                 f'{select_settings if args.select else ""}' \
                 f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
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
    model = MVDet(train_set, args.arch, args.use_bottleneck, args.aggregation, args.hidden_dim, args.outfeat_dim).cuda()

    # load checkpoint
    if args.select:
        with open(f'logs/{args.dataset}/{args.arch}_performance.txt', 'r') as fp:
            result_str = fp.read()
        print(result_str)
        load_dir = result_str.split('\n')[1].replace('# ', '')
        pretrained_dict = torch.load(f'{load_dir}/MultiviewDetector.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'cam_' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth')
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
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    optimizer_cam = optim.Adam(model.cam_pred.parameters(), lr=args.lr * args.select_lr_ratio,
                               weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=int(0.3 * args.epochs)):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)
    # scheduler_cam = torch.optim.lr_scheduler.LambdaLR(optimizer_cam, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args)

    def test_with_select(dataloader, override=None):
        t0 = time.time()
        losses, modas = [], []
        for init_cam in np.arange(train_set.num_cam) if args.select or override is not None else [None]:
            print(f'init camera {init_cam}:')
            init_cam = init_cam * torch.ones([args.batch_size]).long().cuda() if init_cam is not None else None
            loss, moda = trainer.test(None, dataloader, res_fpath, init_cam, override)
            losses.append(loss)
            modas.append(moda)
        loss, moda = np.average(losses), np.average(modas)
        print(f'average moda: {moda:.2f}%, average loss: {loss:.6f}, time: {time.time() - t0:.2f}')
        if override is not None:
            return losses, modas
        return loss, moda

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []

    # learn
    res_fpath = os.path.join(logdir, 'test.txt')

    if not args.eval:
        # trainer.test(args.epochs, test_loader, res_fpath)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            if args.select and not (args.base_lr_ratio or args.base_lr_ratio):
                model.eval()
                model.cam_pred.train()
                train_loss = trainer.train(epoch, train_loader, optimizer_cam,
                                           hard=args.hard, identity_goal=epoch < 0)
            else:
                model.train()
                train_loss = trainer.train(epoch, train_loader, optimizer, scheduler,
                                           hard=args.hard, identity_goal=epoch < 0)
            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, moda = test_with_select(test_loader)

                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                test_loss_s.append(test_loss)
                test_moda_s.append(moda)
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    print('Test loaded model...')
    print(logdir)
    if not args.select and not args.eval:
        val_losses, test_losses = [], []
        val_modas, test_modas = [], []
        for cam in range(train_set.num_cam):
            val_loss, val_moda = test_with_select(val_loader, override=cam)
            val_losses.append(val_loss)
            val_modas.append(val_moda)
            test_loss, test_moda = test_with_select(test_loader, override=cam)
            test_losses.append(test_loss)
            test_modas.append(test_moda)
        val_losses, test_losses = np.array(val_losses), np.array(test_losses)
        val_modas, test_modas = np.array(val_modas), np.array(test_modas)
        loss_strategy = np.argmin(val_losses, axis=1)
        loss_strategy_modas = test_modas[np.arange(train_set.num_cam), loss_strategy]
        result_strategy = np.argmax(val_modas, axis=1)
        result_strategy_modas = test_modas[np.arange(train_set.num_cam), result_strategy]
        best2cam, avg2cam = np.mean(np.max(test_modas, axis=1)), np.mean(test_modas)
        loss2cam = np.mean(loss_strategy_modas)
        result2cam = np.mean(result_strategy_modas)
        _, moda = trainer.test(args.epochs, test_loader, res_fpath)
        np.savetxt(f'{logdir}/losses_val_test.txt', np.concatenate([val_losses, test_losses]), '%.2f')
        fname = f'modas_{moda:.2f}_Lstrategy{loss2cam:.2f}_Rstrategy{result2cam:.2f}_theory{best2cam:.2f}.txt'
        np.savetxt(f'{logdir}/{fname}',
                   np.concatenate([val_modas, test_modas]), '%.2f',
                   header=f'loading checkpoint...\n'
                          f'{logdir}\n'
                          f'val / test',
                   footer=f'\tloss strategy\n' +
                          ' '.join(f'cam {loss_strategy[cam]} {loss_strategy_modas[cam]:.2f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'\tresult strategy\n' +
                          ' '.join(f'cam {result_strategy[cam]} {result_strategy_modas[cam]:.2f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'2 best cam: loss_strategy {loss2cam:.2f}, '
                          f'result_strategy {result2cam:.2f}, theory {best2cam:.2f}\n'
                          f'all cam: {moda:.2f}')
        with open(f'{logdir}/{fname}', 'r') as fp:
            print(fp.read())
        if args.resume is None:
            shutil.copyfile(f'{logdir}/{fname}', f'logs/{args.dataset}/{args.arch}_performance.txt')

    else:
        trainer.test(args.epochs, test_loader, res_fpath)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    # parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=None)
    parser.add_argument('--select_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--select', type=str2bool, default=False)

    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    parser.add_argument('--hard', type=str2bool, default=False)
    parser.add_argument('--beta_conf', type=float, default=0.0)
    parser.add_argument('--beta_even', type=float, default=0.0)

    args = parser.parse_args()

    main(args)
