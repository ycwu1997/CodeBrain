import torch
import torch.nn as nn
from lib.nafnet import encoder, decoder, decoder_2
from lib.finite_scalar_quantization import FSQ
import argparse
import numpy as np
import torch.nn.functional as F

class CodeGenerater(nn.Module):
    def __init__(self, args, opt, code_dim):
        super(CodeGenerater, self).__init__()
        self.code_dim = code_dim
        self.codebooks=FSQ(levels=args.fsq_levels, dim=self.code_dim)
        self.codebook_size = np.prod(args.fsq_levels)

    def forward(self, x):
        q_x, idx_x = self.codebooks(x)
        return q_x, idx_x

class CodeBrain(nn.Module):
    def __init__(self, args, opt):
        super(CodeBrain, self).__init__()
        self.code_dim = len(args.fsq_levels)

        self.posterior_encoder = encoder(input_channels=opt.INPUT_C, output_channels=self.code_dim, naf_dim=args.naf_dim, naf_depth=args.naf_depth)
        self.codebook = CodeGenerater(args, opt, self.code_dim)

        base_dim = args.naf_dim*3
        self.prior_encoder = encoder(input_channels=opt.INPUT_C, naf_dim=base_dim, naf_depth=args.naf_depth)
        self.decoder = decoder_2(code_dim=self.code_dim, output_channels=opt.OUTPUT_C, naf_dim=base_dim, naf_depth=args.naf_depth)

    def forward(self, prior, posterior):   
        encoded_posteriors = self.posterior_encoder(posterior)
        code, indices = self.codebook(encoded_posteriors)

        encoded_priors = self.prior_encoder(prior)
        decoded_imgs = self.decoder(code, encoded_priors)
        return decoded_imgs, indices

class CodeBrain_Grader(nn.Module):
    def __init__(self, args, opt):
        super(CodeBrain_Grader, self).__init__()
        self.fsq_levels = args.fsq_levels
        self.levels = [level - 1 for level in self.fsq_levels]
        self.out_channels = np.sum(self.levels)

        base_dim = args.naf_dim*3
        self.grader = encoder(input_channels=opt.INPUT_C, naf_dim=base_dim, naf_depth=args.naf_depth)
        self.head = nn.Conv2d(8 * base_dim, self.out_channels, kernel_size=1, bias=True)

    def forward(self, prior):
        feat = self.grader(prior)
        logits = self.head(feat)
        logits = torch.split(logits, self.levels, dim=1)
        return logits