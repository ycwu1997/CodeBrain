import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.codebrain import CodeBrain, CodeBrain_Grader
from utils.metrics_set import *
from utils.losses import PSNRLoss
import os
from utils.utils import scale_to_oridinal_labels, scale_to_codes, scale_to_cls_labels, masking_img
from torchmetrics.functional.image import peak_signal_noise_ratio
from torch.nn.parallel import DistributedDataParallel as DDP

class Grader(nn.Module):
    def __init__(self, args, opt, device):
        super(Grader, self).__init__()
        self.device = device
        self.lr = args.lr
        self.fsq_levels = args.fsq_levels

        self.rec_model = CodeBrain(args, opt).to(device)
        ckpt = torch.load(os.path.join(args.rec_path, '{}_{}_best.pth'.format(args.rec_name, opt.DATASET)), weights_only=True, map_location=device)
        self.rec_model.load_state_dict(ckpt['rec_model'])
        self.rec_model.eval()
        
        self.grad_model = CodeBrain_Grader(args, opt).to(device)
        self.opt_grad = torch.optim.AdamW(self.grad_model.parameters(), lr=self.lr, betas=(0.9, 0.9), weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_grad, T_max=args.total_iterations, eta_min=5e-5)

        self.grad_loss = nn.BCEWithLogitsLoss()

        # Resume
        if args.RESUME:
            ckpt = torch.load(args.RESUME_PATH + '/{}_{}_latest.pth'.format(args.grad_name, opt.DATASET), map_location=device)
            self.rec_model.load_state_dict(ckpt['rec_model'])
            self.grad_model.load_state_dict(ckpt['grad_model'])
            self.opt_grad.load_state_dict(ckpt['opt_grad'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.epoch_start = ckpt['epoch_start'] + 1
        else:
            self.epoch_start = 0

    @staticmethod
    def _unwrap_module(module):
        return module.module if isinstance(module, DDP) else module

    def wrap_distributed(self, local_rank):
        self.grad_model = DDP(self.grad_model, device_ids=[local_rank], output_device=local_rank)

    def rec_model_state_dict(self):
        return self.rec_model.state_dict()

    def grad_model_state_dict(self):
        return self._unwrap_module(self.grad_model).state_dict()

    def get_q_codes(self, posteriors):
        with torch.no_grad():
            encoder_feat = self.rec_model.posterior_encoder(posteriors)
            q_x, _ = self.rec_model.codebook(encoder_feat)
        return q_x

    def get_decoded_imgs(self, codes_quant, x_z):
        with torch.no_grad():
            encoded_priors = self.rec_model.prior_encoder(x_z)
            decoded_imgs = self.rec_model.decoder(codes_quant, encoded_priors)
        return decoded_imgs

    def train_step(self, x_r, select_list):
        tensor_list = torch.as_tensor(select_list, device=x_r.device, dtype=x_r.dtype)
        mask_tensor = tensor_list[None, :, None, None]

        self.grad_model.train()
        x_z = x_r * mask_tensor

        q_codes = self.get_q_codes(x_r)
        b, c_token, h_token, w_token = q_codes.shape

        out_levels = self.grad_model(x_z)

        grad_loss, acc, count_grad = 0, 0, 0
        codes_quant = []
        for j, o in enumerate(out_levels):
            g = scale_to_oridinal_labels(q_codes[:, j, :, :], self.fsq_levels[j])
            grad_loss += self.grad_loss(o, g)
            count_grad += 1
            prob_o = torch.sigmoid(o)
            pred_index = (prob_o > 0.5).sum(dim=1)
            codes_quant.append(scale_to_codes(pred_index, self.fsq_levels[j]))

        codes_quant = torch.stack(codes_quant, dim=1)
        loss = grad_loss / count_grad
        acc = (codes_quant == q_codes).sum() / (b*c_token*h_token*w_token)

        self.opt_grad.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_grad.step()
        self.scheduler.step()

        decoded_images = self.get_decoded_imgs(codes_quant, x_z)
        psnr_all = peak_signal_noise_ratio(decoded_images.clamp(0, 1), x_r, data_range=1.0, reduction='none', dim=(0, 2, 3))
        psnr_imp = (psnr_all * (1 - tensor_list)).sum() / (1 - tensor_list).sum()

        loss_dict = {
            'loss': loss.item(),
            'acc': acc,
            'psnr': psnr_imp.item()
        }
        return loss_dict