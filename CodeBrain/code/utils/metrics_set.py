import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional import mean_squared_error, mean_absolute_error

def get_metrics(preds, gts):
    assert preds.shape == gts.shape
    psnr_case = peak_signal_noise_ratio(preds, gts, data_range=1.0)
    ssim_case = structural_similarity_index_measure(preds, gts, data_range=1.0)
    mae_case = mean_absolute_error(preds, gts)
    return psnr_case, ssim_case*100, mae_case*1000