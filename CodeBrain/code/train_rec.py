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
from utils.utils import rand_seed, check_is_legal, Logger
from lib.reconstructor import Reconstructor
from evaluate_rec import validate_rec
from utils.dataset import BaseDataSets, Augmentation

def pre_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/params_ixi.yaml', help='config path (*.yaml)')
    parser.add_argument('--rec_name', type=str, default='cb_rec')

    parser.add_argument('--total_iterations', type=int, default=300000) 
    # we change the training indicator to iterations instead of epochs.
    # the performance will be better with more iterations.
    # also some Transofmer-based methods needs more iterations to converge.
    # for comparisons, please set the same total_iteration for all methods.
    parser.add_argument('--bs', type=int, default=8, help='batchsize')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning_rate')
    # we use a higher LR since the training is stable if without the final tanh activation in the decoder

    parser.add_argument('--naf_dim', type=int, default=32, help='NAF dimension (default: 32)')
    parser.add_argument('--naf_depth', type=int, default=6, help='NAF depth (default: 6), totalling 27 blocks')
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8,8,8,5,5,5])

    parser.add_argument('--r_weight', type=float, default=5, help='Reconstruction loss scalar for available sequences (default: 5)')
    parser.add_argument('--i_weight', type=float, default=20, help='Imputation loss scalar for missing sequences  (default: 20)')
    parser.add_argument('--g_weight', type=float, default=0.1, help='Gan loss scalar (default: 0.1)')

    parser.add_argument('--RESUME', action='store_true') # default FALSE
    parser.add_argument('--RESUME_PATH', type=str, default='../models/cb_rec_IXI_20260318-112206')

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
        opt.MODEL_DIR += args.rec_name + '_{}_{}'.format(opt.DATASET, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(opt.MODEL_DIR, exist_ok=True)
        logger = Logger(args.rec_name, path=opt.MODEL_DIR)
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

    rec_trainer = Reconstructor(args, opt, device)
    rec_trainer.wrap_distributed(local_rank)
    epoch_start = rec_trainer.epoch_start

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
            Loss_dict = rec_trainer.train_step(images, select_list, iterations)

            if master_process:
                if step % 10 == 0:
                    perplexity = Loss_dict['perplexity']
                    used_percentage = Loss_dict['used_percentage']

                    writer.add_scalar('Codebook/perplexity', perplexity, iterations)
                    writer.add_scalar('Codebook/used_percentage', used_percentage, iterations)
                    writer.add_scalar('PSNR/psnr_rec', Loss_dict['psnr_rec'], iterations)
                    writer.add_scalar('PSNR/psnr_imp', Loss_dict['psnr_imp'], iterations)
                    writer.add_scalar('R_Loss/total_rec', Loss_dict['total_rec'], iterations)
                    writer.add_scalar('R_Loss/loss_rec', Loss_dict['rec_loss'], iterations)
                    writer.add_scalar('R_Loss/loss_imp', Loss_dict['imp_loss'], iterations)
                    writer.add_scalar('R_Loss/loss_gan', Loss_dict['gan_loss'], iterations)
                    writer.add_scalar('D_Loss/Disc', Loss_dict['total_disc'], iterations)

                    print_str = "PSNR_Imp: {:.5f} dB, PSNR_Rec: {:.5f} dB, Imp Loss: {:.5f}, GAN Loss: {:.5f}, Rec Loss: {:.5f}, Used Percentage: {:.5f}%, Perplexity: {:.5f}.".format(Loss_dict['psnr_imp'], Loss_dict['psnr_rec'], Loss_dict['imp_loss'], Loss_dict['gan_loss'], Loss_dict['rec_loss'], used_percentage, perplexity)
                    logger.write_and_print(print_str)
                
            should_validate = iterations == 1000 or iterations % 10000 == 0 or (iterations > 0.8 * args.total_iterations and iterations % 2000 == 0)
            if should_validate:
                if dist.is_initialized():
                    dist.barrier()
                if master_process:
                    metric_instance_ = validate_rec(rec_trainer, val_loader, opt, writer, iterations)
                    logger.write_and_print('Overall PSNR: {}'.format(metric_instance_))

                    if metric_instance_ >= best_metric:
                        best_metric = metric_instance_
                        logger.write_and_print('Best PSNR: {}'.format(best_metric))
                        
                        ckpt = {'rec_model': rec_trainer.rec_model_state_dict()}
                        torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_best.pth'.format(args.rec_name, opt.DATASET)))
                        torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_{}_best.pth'.format(args.rec_name, opt.DATASET, iterations, best_metric)))
                if dist.is_initialized():
                    dist.barrier()
                    
        if master_process:
            ckpt = {'rec_model': rec_trainer.rec_model_state_dict(),
                    'gan': rec_trainer.gan_state_dict(),
                    'opt_rec': rec_trainer.opt_rec.state_dict(),
                    'opt_disc': rec_trainer.opt_disc.state_dict(),
                    'scheduler': rec_trainer.scheduler.state_dict(),
                    'epoch_start': epoch
                    }
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_latest.pth'.format(args.rec_name, opt.DATASET)))

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
