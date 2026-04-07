import torch
import torch.nn as nn
import numpy as np
import random
from scipy.ndimage import zoom
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import logging
import os
import time

class Logger:
    def __init__(self, model_name, path='./log/'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        current_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.log_path = path
        os.makedirs(self.log_path, exist_ok=True)
        assert os.path.exists(self.log_path), 'Make sure the log path is exists.'
        self.log_file_name = os.path.join(self.log_path, model_name+'_'+current_time+'.log')
        self.file_handle = logging.FileHandler(self.log_file_name, mode='w')
        self.file_handle.setLevel(logging.DEBUG)

        # set logger format
        formatter = logging.Formatter("%(message)s")
        self.file_handle.setFormatter(formatter)

        self.logger.addHandler(self.file_handle)

    def write(self, message):
        self.logger.info(message)

    def write_and_print(self, message):
        self.logger.info(message)
        print(message)

def norm_img(x, eps=1e-8):
    x = (x - x.min()) / ((x.max() - x.min()+eps))
    return x

def rand_seed(SEED=1234):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def show_img(volumes, preds):
    bs, c_num, width, height = volumes.size()
    concat_pred = np.zeros([3*width, height * 2 * c_num])

    for idx in range(c_num):
        volume_slice = volumes[0, idx].float().numpy()
        pred_slice = preds[0, idx].numpy()
        diff_slice = norm_img(volume_slice - pred_slice)
        col_start = 2 * idx * height
        hist_col_start = col_start + height
        
        concat_pred[:width, col_start:col_start + height] = volume_slice
        concat_pred[width:2*width, col_start:col_start + height] = pred_slice
        concat_pred[2*width:3*width, col_start:col_start + height] = diff_slice
        concat_pred[:width, hist_col_start:hist_col_start + height] = plot_hist(volume_slice)
        concat_pred[width:2*width, hist_col_start:hist_col_start + height] = plot_hist(pred_slice)
        concat_pred[2*width:3*width, hist_col_start:hist_col_start + height] = plot_hist(diff_slice)
    return concat_pred

def plot_hist(slice):
    H, W = slice.shape
    fig, ax = plt.subplots(figsize=(8, 6))

    hist, bins = np.histogram(slice.flatten(), bins=256, range=(0, 1))
    ax.bar(bins[:-1], hist, width=1/256, alpha=0.7, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(hist) * 1.1)
    ax.grid(True, alpha=0.3)
    
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    gray_buf = np.dot(buf[..., :3], [0.299, 0.587, 0.114])
    
    zoom_factor_h = H / gray_buf.shape[0]
    zoom_factor_w = W / gray_buf.shape[1]
    histogram_matrix = zoom(gray_buf, (zoom_factor_h, zoom_factor_w), order=1)
    
    histogram_matrix = (histogram_matrix - np.min(histogram_matrix)) / (np.max(histogram_matrix) - np.min(histogram_matrix) + 1e-8)
    
    plt.close(fig)
    
    return histogram_matrix

def scale_to_oridinal_labels(y, s):
    assert s > 1
    denom = s if (s % 2 == 0) else (s - 1)
    y = (y + 1) * 0.5 * denom 
    y = y.unsqueeze(1)
    thresholds = torch.arange(s-1, device = y.device).view(1, -1, 1, 1)
    ordinal_labels = (y > thresholds).float()
    return ordinal_labels

def scale_to_cls_labels(y, s):
    assert s > 1
    denom = s if (s % 2 == 0) else (s - 1)
    y = (y + 1) * 0.5 * denom
    y = y.unsqueeze(1).long()
    return y

def scale_to_codes(y, s):
    assert s > 1
    denom = s if (s % 2 == 0) else (s - 1)
    y = y / denom * 2 - 1
    return y

def check_is_legal(inp):
    assert not torch.isnan(inp).any(), "Input contains NaN!"
    assert not torch.isinf(inp).any(), "Input contains Inf!"

def masking_img(images, list = [0,1,1]):
    assert len(images.shape) == 4 # b, c, h, w
    assert images.shape[1] == len(list) # c = 3 or 4
    mask_tensor = torch.as_tensor(list, device=images.device, dtype=images.dtype)[None, :, None, None]
    return images * mask_tensor
