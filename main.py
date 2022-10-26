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

    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size,
                             dropout=args.dropcam, augmentation=args.augmentation)
    val_set = frameDataset(base, split='val', world_reduce=args.world_reduce,
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
        select_settings = f'hard{int(args.hard)}_conf{args.beta_conf}even{args.beta_even}cover{args.beta_coverage}_'
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

    def warmup_lr_scheduler(epoch, warmup_epochs=int(0.3 * args.epochs)):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args)

    def test_with_select(dataloader, override=None, result_type='moda'):
        t0 = time.time()
        losses, precs = [], []
        for init_cam in range(train_set.num_cam) if args.select or override is not None else [None]:
            print(f'init camera {init_cam}:')
            loss, prec = trainer.test(dataloader, init_cam, override)
            losses.append(loss)
            precs.append(prec)
        loss, prec = np.average(losses), np.average(precs)
        print(f'average {result_type}: {prec:.1f}%, average loss: {loss:.6f}, time: {time.time() - t0:.1f}')
        if override is not None:
            return losses, precs
        return loss, prec

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_prec_s = []

    # learn
    if not args.eval:
        # trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer.train(epoch, train_loader, optimizer, scheduler, hard=args.hard, )
            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, test_prec = test_with_select(test_loader)

                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec)
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                           test_result=test_prec_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    print('Test loaded model...')
    print(logdir)

    def log_best2cam_strategy(result_type='moda'):
        val_losses, val_precs = [], []
        test_losses, test_precs = [], []
        for cam in range(train_set.num_cam):
            val_loss, val_prec = test_with_select(val_loader, override=cam, result_type=result_type)
            val_losses.append(val_loss)
            val_precs.append(val_prec)
            test_loss, test_prec = test_with_select(test_loader, override=cam, result_type=result_type)
            test_losses.append(test_loss)
            test_precs.append(test_prec)
        val_losses, val_precs = np.array(val_losses), np.array(val_precs)
        test_losses, test_precs = np.array(test_losses), np.array(test_precs)
        loss_strategy = np.argmin(val_losses, axis=1)
        loss_strategy_precs = test_precs[np.arange(train_set.num_cam), loss_strategy]
        loss2cam = np.mean(loss_strategy_precs)
        result_strategy = np.argmax(val_precs, axis=1)
        result_strategy_precs = test_precs[np.arange(train_set.num_cam), result_strategy]
        result2cam = np.mean(result_strategy_precs)
        theory_strategy = np.argmax(test_precs, axis=1)
        theory_strategy_precs = test_precs[np.arange(train_set.num_cam), theory_strategy]
        best2cam = np.mean(theory_strategy_precs)
        _, prec = trainer.test(test_loader)
        np.savetxt(f'{logdir}/losses_val_test.txt', np.concatenate([val_losses, test_losses]), '%.2f')
        fname = f'{result_type}s_{prec:.1f}_Lstrategy{loss2cam:.1f}_Rstrategy{result2cam:.1f}_theory{best2cam:.1f}.txt'
        np.savetxt(f'{logdir}/{fname}',
                   np.concatenate([val_precs, test_precs]), '%.1f',
                   header=f'loading checkpoint...\n'
                          f'{logdir}\n'
                          f'val / test',
                   footer=f'\tloss strategy\n' +
                          ' '.join(f'cam {loss_strategy[cam]} {loss_strategy_precs[cam]:.1f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'\tresult strategy\n' +
                          ' '.join(f'cam {result_strategy[cam]} {result_strategy_precs[cam]:.1f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'\ttheory\n' +
                          ' '.join(f'cam {theory_strategy[cam]} {theory_strategy_precs[cam]:.1f} |'
                                   for cam in range(train_set.num_cam)) + '\n' +
                          f'2 best cam: loss_strategy {loss2cam:.1f}, '
                          f'result_strategy {result2cam:.1f}, theory {best2cam:.1f}\n'
                          f'all cam: {prec:.1f}')
        with open(f'{logdir}/{fname}', 'r') as fp:
            print(fp.read())
        if args.resume is None:
            shutil.copyfile(f'{logdir}/{fname}', f'logs/{args.dataset}/{args.arch}_performance.txt')

    if not args.select and not args.eval:
        log_best2cam_strategy()
    else:
        trainer.test(test_loader)


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
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--select_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
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
    parser.add_argument('--beta_coverage', type=float, default=0.1)

    args = parser.parse_args()

    main(args)
