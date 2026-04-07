import torch
import torch.nn as nn
from lib.discriminator import Discriminator
from lib.codebrain import CodeBrain
from utils.metrics_set import *
from utils.losses import PSNRLoss
from utils.utils import masking_img
from torchmetrics.functional.image import peak_signal_noise_ratio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Reconstructor(nn.Module):
    def __init__(self, args, opt, device):
        super(Reconstructor, self).__init__()
        self.device = device
        self.lr = args.lr

        self.rec_model = CodeBrain(args, opt).to(device)
        self.K = self.rec_model.codebook.codebook_size
        self.gan = Discriminator(in_channels=opt.INPUT_C, out_channels=opt.INPUT_C).to(device)
        
        self.opt_rec = torch.optim.AdamW(self.rec_model.parameters(), lr=self.lr, betas=(0.9, 0.9), weight_decay=0.0)
        self.opt_disc = torch.optim.AdamW(self.gan.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_rec, T_max=args.total_iterations, eta_min=5e-5)
        self.gan_start_iterations = 0.2 * args.total_iterations

        self.rec_loss = PSNRLoss() #nn.L1Loss(reduction='none') 
        # PSNRLoss can achieve better PSNR but slighltly lower SSIM. Following papers, we use PSNRLoss.
        # For other applications, we recommend using L1 with more training iterations. It is more stable.
        self.gan_loss = nn.MSELoss()
        # We use LSGAN following https://github.com/trane293/mm-gan/blob/master/train_mmgan_brats2018.py. 
        # Other GAN losses can be explored. Note that, stronger GAN may reduce the pixel-level performance.
        self.used_probs = None
        self.perplexity = 0
        self.used_percentage = 0
        
        self.rec_weight = args.r_weight
        self.imp_weight = args.i_weight
        self.adv_weight = args.g_weight

        # Resume
        if args.RESUME:
            ckpt = torch.load(args.RESUME_PATH + '/{}_{}_latest.pth'.format(args.rec_name, opt.DATASET), map_location=device)
            self.rec_model.load_state_dict(ckpt['rec_model'])
            self.gan.load_state_dict(ckpt['gan'])
            self.opt_rec.load_state_dict(ckpt['opt_rec'])
            self.opt_disc.load_state_dict(ckpt['opt_disc'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.epoch_start = ckpt['epoch_start'] + 1
        else:
            self.epoch_start = 0

    @staticmethod
    def _unwrap_module(module):
        return module.module if isinstance(module, DDP) else module

    def wrap_distributed(self, local_rank):
        self.rec_model = DDP(self.rec_model, device_ids=[local_rank], output_device=local_rank)
        self.gan = DDP(self.gan, device_ids=[local_rank], output_device=local_rank)

    def rec_model_state_dict(self):
        return self._unwrap_module(self.rec_model).state_dict()

    def gan_state_dict(self):
        return self._unwrap_module(self.gan).state_dict()

    def set_used_indices(self, indices):
        indices = indices.long().detach().view(-1)
        counts = torch.bincount(indices, minlength=self.K).to(torch.float32).to(indices.device)

        if dist.is_initialized() and dist.is_available():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        
        total_counts = torch.sum(counts)
        assert total_counts > 0
        p_inst = counts / total_counts

        # Update moving average of usage as a training indicator.
        momentum = 0.9
        if self.used_probs is None:
            self.used_probs = p_inst
        else:
            self.used_probs = momentum * self.used_probs + (1 - momentum) * p_inst

        self.used_probs = (self.used_probs / self.used_probs.sum().clamp(min=1e-12)).detach()

        p = self.used_probs.clamp(min=1e-12)
        entropy = -(p * p.log()).sum()
        self.perplexity = float(torch.exp(entropy))
        self.used_percentage = float((self.used_probs > 1e-6).float().mean() * 100.0)
    
    def is_train_gan(self, mode=True):
        if mode:
            self.gan.train()
        else:
            self.gan.eval()
        for p in self.gan.parameters():
            p.requires_grad_(mode)

    def train_step(self, x_r, select_list, iterations):
        tensor_list = torch.as_tensor(select_list, device=x_r.device, dtype=x_r.dtype)
        mask_tensor = tensor_list[None, :, None, None]

        self.rec_model.train()
        x_z = x_r * mask_tensor

        fake_x, indices = self.rec_model(x_z, x_r)
        self.set_used_indices(indices)
        fake_x_masked = fake_x * (1-mask_tensor) + x_z

        # for training Discriminator
        self.is_train_gan(True)
        pred_real = self.gan(x_r, x_r)
        loss_real = self.gan_loss(pred_real, torch.ones_like(pred_real))

        pred_fake = self.gan(fake_x_masked.detach(), x_r)
        loss_fake = self.gan_loss(pred_fake, torch.ones_like(pred_fake)*mask_tensor)

        total_disc_loss = 0.5 * (loss_real + loss_fake)

        self.opt_disc.zero_grad(set_to_none=True)
        total_disc_loss.backward()
        self.opt_disc.step()

        # for training Reconstructor
        loss_all = self.rec_loss(fake_x, x_r)

        mask_count = tensor_list.sum()
        imp_count = (1 - tensor_list).sum()

        loss_rec = (loss_all * tensor_list).sum() / mask_count
        loss_imp = (loss_all * (1 - tensor_list)).sum() / imp_count

        total_rec_loss = self.rec_weight * loss_rec + self.imp_weight * loss_imp

        self.is_train_gan(False)
        pred_fake2 = self.gan(fake_x_masked, x_r)
        loss_gan = self.gan_loss(pred_fake2, torch.ones_like(pred_fake2))
        if iterations > self.gan_start_iterations:
            total_rec_loss += self.adv_weight * loss_gan
            disc_factor = 1
        else:
            disc_factor = 0

        self.opt_rec.zero_grad(set_to_none=True)
        total_rec_loss.backward()
        self.opt_rec.step()
        self.scheduler.step()

        loss_dict = {
            'total_rec': total_rec_loss.item(),
            'psnr_imp': -loss_imp.item(),
            'psnr_rec': -loss_rec.item(),
            'imp_loss': self.imp_weight * loss_imp.item(),
            'gan_loss': self.adv_weight * loss_gan.item() * disc_factor,
            'rec_loss': self.rec_weight * loss_rec.item(),
            'total_disc': total_disc_loss.item(),
            'perplexity': self.perplexity,
            'used_percentage': self.used_percentage
        }
        return loss_dict