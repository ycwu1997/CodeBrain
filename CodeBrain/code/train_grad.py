import os
import argparse
import datetime
import shutil
import random

import torch
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm
from tensorboardX import SummaryWriter

from configs.config import *
from utils.utils import rand_seed, Logger, check_is_legal
from lib.grader import Grader
from evaluate_grad import validate_grad
from utils.dataset import BaseDataSets, Augmentation

def pre_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/params_ixi.yaml', help='config path (*.yaml)')
    parser.add_argument('--grad_name', type=str, default='cb_grad')

    parser.add_argument('--total_iterations', type=int, default=300000) 
    # we change the training indicator to iterations instead of epochs.
    # the performance will be better with more iterations.
    # also some Transofmer-based methods needs more iterations to converge.
    # for comparisons, please set the same total_iteration for all methods.
    parser.add_argument('--bs', type=int, default=16, help='batchsize')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning_rate')
    # we use a higher LR since the training is stable if without the final tanh activation in the decoder

    parser.add_argument('--naf_dim', type=int, default=32, help='NAF dimension (default: 32)')
    parser.add_argument('--naf_depth', type=int, default=6, help='NAF depth (default: 6), totalling 27 blocks')
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8,8,8,5,5,5])

    parser.add_argument('--rec_path', type=str, default='../models/cb_rec_IXI_20260318-112206')
    parser.add_argument('--rec_name', type=str, default='cb_rec')
    
    parser.add_argument('--RESUME', action='store_true') # default FALSE
    parser.add_argument('--RESUME_PATH', type=str, default='../models/cb_grad_IXI_20250405-175703')

    args = parser.parse_args()
    opt = Config(config_path=args.config)
    rand_seed(opt.RANDOM_SEED)
    return args, opt


def main(args, opt):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    try:
        dist.init_process_group('nccl', device_id=device)
    except TypeError:
        dist.init_process_group('nccl')

    master_process = local_rank == 0
    if master_process:
        opt.MODEL_DIR += args.grad_name + '_{}_{}'.format(opt.DATASET, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(opt.MODEL_DIR, exist_ok=True)
        logger = Logger(args.grad_name, path=opt.MODEL_DIR)
        writer = SummaryWriter(opt.MODEL_DIR)
        shutil.copytree('../code/', opt.MODEL_DIR + '/code/', shutil.ignore_patterns(['.git','__pycache__']))

    # dataset
    db_train = BaseDataSets(
        base_dir=opt.DATA_PATH,
        dataset=opt.DATASET,
        modality_list = opt.MODALITY_LIST,
        split='train'
    )
    db_val = BaseDataSets(
        base_dir=opt.DATA_PATH,
        dataset=opt.DATASET,
        modality_list = opt.MODALITY_LIST,
        split='val'
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
    train_loader = DataLoader(db_train, 
                              batch_size=args.bs, 
                              num_workers=opt.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(db_val, batch_size=4, num_workers=1, pin_memory=True, shuffle=False)

    total_epochs = args.total_iterations // len(train_loader)
    best_metric = 0

    grad_trainer = Grader(args, opt, device)
    grad_trainer.wrap_distributed(local_rank)
    epoch_start = grad_trainer.epoch_start

    RandomAug = Augmentation(device)

    for epoch in range(epoch_start, total_epochs):
        train_sampler.set_epoch(epoch)
        if master_process:
            print_str = '-------epoch {}/{}-------'.format(epoch, total_epochs)
            logger.write_and_print(print_str)
        
        for step, sample in enumerate(tqdm(train_loader, disable=not master_process)):
            # prepare data
            images = sample['images'].to(device)
            images = RandomAug(images)
            # apply random augmentation to the images can improve the performance slightly, here we make it optional
            # check_is_legal(images)

            iterations = len(train_loader) * epoch + step + 1
            select_list = random.choice(opt.IMPUTE_LIST)
            loss_dict = grad_trainer.train_step(images, select_list)

            if master_process:
                if step % 10 == 0:
                    writer.add_scalar('Loss/grad', loss_dict['loss'], iterations)
                    writer.add_scalar('Metrics/psnr', loss_dict['psnr'], iterations)
                    writer.add_scalar('Metrics/acc', loss_dict['acc'], iterations)

                    print_str = "Training PSNR: {:.5f} dB, Acc: {:.5f}, Loss_grad: {:.5f}".format(loss_dict['psnr'], loss_dict['acc'], loss_dict['loss'])
                    logger.write_and_print(print_str)

            should_validate = iterations == 1000 or iterations % 10000 == 0 or (iterations > 0.8 * args.total_iterations and iterations % 2000 == 0)
            if should_validate:
                if dist.is_initialized():
                    dist.barrier()
                if master_process:
                    metric_instance_ = validate_grad(grad_trainer, val_loader, opt, writer, iterations)
                    logger.write_and_print('Overall PSNR: {}'.format(metric_instance_))
                    
                    if metric_instance_ >= best_metric:
                        best_metric = metric_instance_
                        logger.write_and_print('Best PSNR: {}'.format(best_metric))

                        ckpt = {'rec_model': grad_trainer.rec_model_state_dict(),
                                'grad_model': grad_trainer.grad_model_state_dict()
                                }
                        torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_best.pth'.format(args.grad_name, opt.DATASET)))
                        torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_{}_best.pth'.format(args.grad_name, opt.DATASET, iterations, best_metric)))
                if dist.is_initialized():
                    dist.barrier()

        if master_process:
            ckpt = {'rec_model': grad_trainer.rec_model_state_dict(),
                    'grad_model': grad_trainer.grad_model_state_dict(),
                    'opt_grad': grad_trainer.opt_grad.state_dict(),
                    'scheduler': grad_trainer.scheduler.state_dict(),
                    'epoch_start': epoch
                    }
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_latest.pth'.format(args.grad_name, opt.DATASET)))

    if dist.is_initialized():
        dist.destroy_process_group()
        print("Training finished, process group destroyed, exiting safely.")

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    args, opt= pre_setup()
    try:
        main(args, opt)
    except KeyboardInterrupt:
        if dist.is_initialized():
            dist.destroy_process_group()
        print("Process group destroyed, exit safely.")
