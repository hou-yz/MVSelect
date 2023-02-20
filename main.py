import os
import time
import itertools

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
from src.datasets import *
from src.models.mvdet import MVDet
from src.models.mvcnn import MVCNN
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer, find_dataset_lvl_strategy
from src.trainer_mvcnn import ClassifierTrainer


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
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
    if 'modelnet' in args.dataset:
        if args.dataset == 'modelnet40_12':
            fpath = os.path.expanduser('~/Data/modelnet/modelnet40_images_new_12x')
            num_cam = 12
        elif args.dataset == 'modelnet40_20':
            fpath = os.path.expanduser('~/Data/modelnet/modelnet40v2png_ori4')
            num_cam = 20
        else:
            raise Exception

        args.task = 'mvcnn'
        result_type = ['prec']
        args.lr = 5e-5 if args.lr is None else args.lr
        args.select_lr = 2e-5 if args.select_lr is None else args.select_lr
        args.batch_size = 8 if args.batch_size is None else args.batch_size

        train_set = imgDataset(fpath, num_cam, split='train', )
        val_set = imgDataset(fpath, num_cam, split='train', per_cls_instances=25)
        test_set = imgDataset(fpath, num_cam, split='test', )
    else:
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise Exception('must choose from [wildtrack, multiviewx]')

        args.task = 'mvdet'
        result_type = ['moda', 'modp', 'prec', 'recall']
        args.lr = 5e-4 if args.lr is None else args.lr
        args.select_lr = 1e-3 if args.select_lr is None else args.select_lr
        args.batch_size = 1 if args.batch_size is None else args.batch_size

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

    if args.steps:
        args.lr /= 5
        args.epochs *= 2

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
    N = train_set.num_cam

    # logging
    select_settings = f'steps{args.steps}_entropy{args.beta_entropy}_'
    lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}' + \
                  f'select{args.select_lr}' if args.steps else ''
    logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}{args.arch}_{args.aggregation}_down{args.down}_' \
             f'{select_settings if args.steps else ""}' \
             f'lr{args.lr}{lr_settings}_b{args.batch_size}_e{args.epochs}_dropcam{args.dropcam}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', logdir + '/scripts/src')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    if args.task == 'mvcnn':
        model = MVCNN(train_set, args.arch, args.aggregation).cuda()
    else:
        model = MVDet(train_set, args.arch, args.aggregation,
                      args.use_bottleneck, args.hidden_dim, args.outfeat_dim).cuda()

    # load checkpoint
    if args.steps:
        with open(f'logs/{args.dataset}/{args.arch}_performance.txt', 'r') as fp:
            result_str = fp.read()
        print(result_str)
        load_dir = result_str.split('\n')[1].replace('# ', '')
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'select' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'select' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'select' in n and p.requires_grad],
                    "lr": args.select_lr, }, ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    if args.task == 'mvcnn':
        trainer = ClassifierTrainer(model, logdir, args)
    else:
        trainer = PerspectiveTrainer(model, logdir, args)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    # learn
    if not args.eval:
        # trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler)
            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, test_prec = trainer.test(test_loader, torch.eye(N) if args.steps else None)

                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec[0])
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                           train_prec_s, test_prec_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    def log_best2cam_strategy(result_type=('prec',), max_steps=4):
        candidates = np.eye(N)
        combinations = np.array(list(itertools.combinations(candidates, 2))).sum(1)
        combination_indices = np.array(list(itertools.combinations(list(range(N)), 2)))
        # diagonal: step == 0
        val_loss_diag, val_prec_diag, _ = trainer.test_cam_combination(val_loader, candidates)
        test_loss_diag, test_prec_diag, _ = trainer.test_cam_combination(test_loader, candidates)
        # non-diagonal: step == 1
        val_loss_s, val_prec_s, val_oracle_s = trainer.test_cam_combination(val_loader, combinations)
        test_loss_s, test_prec_s, test_oracle_s = trainer.test_cam_combination(test_loader, combinations)
        for i in range(2, max_steps):
            trainer.test_cam_combination(test_loader, np.array(list(itertools.combinations(candidates, i + 1))).sum(1))

        def combine2mat(diag_terms, non_diag_terms):
            combined_mat = np.zeros([len(diag_terms), len(diag_terms)] + list(diag_terms.shape[1:]))
            combined_mat[np.eye(len(diag_terms), dtype=bool)] = diag_terms
            non_diag_indices = list(itertools.combinations(list(range(len(diag_terms))), 2))
            for i in range(len(non_diag_indices)):
                idx = non_diag_indices[i]
                combined_mat[idx[0], idx[1]] = combined_mat[idx[1], idx[0]] = non_diag_terms[i]
            return combined_mat

        def find_cam(init_cam, combination_id):
            cam_tuple = list(combination_indices[combination_id])
            cam_tuple.remove(init_cam)
            return cam_tuple[0]

        val_loss_strategy = find_dataset_lvl_strategy(-val_loss_s, combinations)
        val_metric_strategy = find_dataset_lvl_strategy(val_prec_s[:, 0], combinations)
        test_metric_strategy = find_dataset_lvl_strategy(test_prec_s[:, 0], combinations)

        _, prec = trainer.test(test_loader)
        np.savetxt(f'{logdir}/losses_val_test.txt', np.concatenate([combine2mat(val_loss_diag, val_loss_s),
                                                                    combine2mat(test_loss_diag, test_loss_s)]), '%.2f')
        for i in range(len(result_type)):
            fname = f'{result_type[i]}_{prec[i]:.1f}_' \
                    f'Lstrategy{test_prec_s[val_loss_strategy].mean(0)[i]:.1f}_' \
                    f'Rstrategy{test_prec_s[val_metric_strategy].mean(0)[i]:.1f}_' \
                    f'theory{test_prec_s[test_metric_strategy].mean(0)[i]:.1f}_' \
                    f'avg{test_prec_s.mean(0)[i]:.1f}.txt'
            np.savetxt(f'{logdir}/{fname}',
                       np.concatenate([combine2mat(val_prec_diag, val_prec_s)[:, :, i],
                                       combine2mat(test_prec_diag, test_prec_s)[:, :, i]]), '%.1f',
                       header=f'loading checkpoint...\n'
                              f'{logdir}\n'
                              f'val / test',
                       footer=f'\tdataset level: loss strategy\n' +
                              ' '.join(f'cam {find_cam(cam, val_loss_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[val_loss_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tdataset level: result strategy\n' +
                              ' '.join(f'cam {find_cam(cam, val_metric_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[val_metric_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tdataset level: theory\n' +
                              ' '.join(f'cam {find_cam(cam, test_metric_strategy[cam])} |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_prec_s[test_metric_strategy][cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'\tinstance level: oracle\n' +
                              ' '.join(f'----- |' for cam in range(N)) + '\n' +
                              ' '.join(f'{test_oracle_s[cam, i]:.1f}% |'
                                       for cam in range(N)) + '\n' +
                              f'2 best cam: loss_strategy {test_prec_s[val_loss_strategy].mean(0)[i]:.1f}, '
                              f'result_strategy {test_prec_s[val_metric_strategy].mean(0)[i]:.1f}, '
                              f'theory {test_prec_s[test_metric_strategy].mean(0)[i]:.1f}, '
                              f'oracle {test_oracle_s.mean(0)[i]:.1f}, average {test_prec_s.mean(0)[i]:.1f}\n'
                              f'all cam: {prec[i]:.1f}')
            with open(f'{logdir}/{fname}', 'r') as fp:
                if i == 0:
                    print(fp.read())
            if args.resume is None and i == 0:
                shutil.copyfile(f'{logdir}/{fname}', f'logs/{args.dataset}/{args.arch}_performance.txt')

    print('Test loaded model...')
    print(logdir)
    if args.steps == 0:
        log_best2cam_strategy(result_type)
    else:
        trainer.test(test_loader)


if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view selection for multiview classification & detection')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                        choices=['wildtrack', 'multiviewx', 'modelnet40_12', 'modelnet40_20'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate for task network')
    parser.add_argument('--select_lr', type=float, default=None, help='learning rate for MVselect')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    # MVSelect settings
    parser.add_argument('--steps', type=int, default=0,
                        help='number of camera views to choose. if 0, then no selection')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor (default: 0.99)')
    parser.add_argument('--down', type=int, default=1, help='down sample the image to 1/N size')
    parser.add_argument('--beta_entropy', type=float, default=0.01)
    # multiview detection specific settings
    parser.add_argument('--eval_init_cam', type=str2bool, default=False,
                        help='only consider pedestrians covered by the initial camera')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
