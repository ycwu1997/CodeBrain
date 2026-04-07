import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F
from configs.config import *
from utils.utils import rand_seed, show_img, scale_to_codes, check_is_legal, masking_img
from utils.metrics_set import *
from utils.dataset import BaseDataSets
from torch.utils.data import DataLoader
from lib.codebrain import CodeBrain, CodeBrain_Grader
import random
from torchmetrics.functional.image import peak_signal_noise_ratio

def validate_grad(Trainer, val_loader, opt, writer, current_iter):
    """
    Validate function for imputation task.
    """
    missing_patterns=opt.IMPUTE_VAL_LIST
    PSNR_dict = {}
    count_dict = {}
    fsq_levels = Trainer.fsq_levels

    for pattern in missing_patterns:
        key = tuple(pattern)
        PSNR_dict[key] = [0.0]*len(opt.MODALITY_LIST)
        count_dict[key] = [0]*len(opt.MODALITY_LIST)

    model = Trainer._unwrap_module(Trainer.grad_model)
    model.eval()

    monitor_step = random.randint(0, len(val_loader)-1) # only check one time
    with torch.no_grad():
        for val_step, sample in enumerate(tqdm(val_loader)):
            volumes = sample['images'].cuda()  # shape: (B, C, H, W)
            # check_is_legal(volumes)
            b, c_modal, h, w = volumes.size()

            for pattern in missing_patterns:
                key = tuple(pattern)

                # Create partial volume by zeroing out missing channels
                masked_volumes = masking_img(volumes, pattern)
                out_levels = model(masked_volumes)

                codes_quant = []
                for j, o in enumerate(out_levels):
                    prob_o = torch.sigmoid(o)
                    pred_index = (prob_o > 0.5).sum(dim=1)
                    codes_quant.append(scale_to_codes(pred_index, fsq_levels[j]))
                codes_quant = torch.stack(codes_quant, dim=1)

                encoded_priors = Trainer.rec_model.prior_encoder(masked_volumes)
                output_volumes = Trainer.rec_model.decoder(codes_quant, encoded_priors)

                for ch in range(c_modal):
                    if pattern[ch] == 0:  # Only evaluate missing modalities
                        gt_slice = volumes[:, ch, :, :].unsqueeze(1)  # (B,1,H,W)  
                        output_slice = output_volumes[:, ch, :, :].unsqueeze(1)
                        output_slice = output_slice.clamp(0, 1)

                        psnr_val = peak_signal_noise_ratio(output_slice, gt_slice, data_range=1.0)

                        PSNR_dict[key][ch] += psnr_val.item()
                        count_dict[key][ch] += 1

                        # Visualization (optional)
                        key_str = str(key).replace("(", "").replace(")", "").replace(", ", "_")
                        if opt.VISUALIZE and val_step == monitor_step:
                            concat_pred = show_img(gt_slice.cpu(), output_slice.cpu())
                            cv2.imshow(f'Pattern_{key_str}_Predictions_{opt.MODALITY_LIST[ch]}', concat_pred)
                            cv2.waitKey(0)

                        # TensorBoard logging (optional)
                        if writer is not None and val_step == monitor_step:
                            concat_pred = show_img(gt_slice.cpu(), output_slice.cpu())
                            writer.add_image(f'Pattern_{key_str}_Modality_{opt.MODALITY_LIST[ch]}', concat_pred, current_iter, dataformats='HW')

    # Compute per-pattern and per-channel means
    final_metrics = {}
    for pattern in missing_patterns:
        key = tuple(pattern)
        c_psnr = []

        for ch in range(len(opt.MODALITY_LIST)):
            n_ch = count_dict[key][ch]
            if n_ch > 0:  # Only consider missing modalities
                c_psnr.append(PSNR_dict[key][ch] / n_ch)
            else:
                # No reconstruction for this channel in this pattern
                c_psnr.append(None)

        valid_psnr = [v for v in c_psnr if v is not None]
        mean_psnr = np.mean(valid_psnr) if valid_psnr else 0

        final_metrics[key] = {
            "PSNR_list": c_psnr,
            "Mean_PSNR": mean_psnr
        }

    # Compute overall averages (across all patterns/channels that were reconstructed)
    overall_psnr_sum = 0.0
    overall_count = 0

    for pattern in missing_patterns:
        key = tuple(pattern)
        for ch in range(len(opt.MODALITY_LIST)):
            if pattern[ch] == 0:  # Only consider missing modalities
                n_ch = count_dict[key][ch]
                if n_ch > 0:
                    overall_psnr_sum += PSNR_dict[key][ch]
                    overall_count += n_ch

    overall_avg_psnr = overall_psnr_sum / overall_count if overall_psnr_sum > 0 else 0

    # Log metrics to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('Validation/Overall_PSNR', overall_avg_psnr, current_iter)
        
        # Log per-pattern metrics
        for pattern_key, metrics in final_metrics.items():
            pattern_str = "_".join(str(v) for v in pattern_key)
            writer.add_scalar(f'Validation/Pattern_{pattern_str}_PSNR', metrics['Mean_PSNR'], current_iter)

    return overall_avg_psnr

def evaluate_grad(rec_model, grad_model, test_loader, opt, result_path, device):
    """
    Evaluate function for reconstruction task.
    """

    missing_patterns = opt.IMPUTE_LIST

    # Accumulators for SSIM, PSNR, MAE
    SSIM_dict = {}
    PSNR_dict = {}
    MAE_dict = {}
    count_dict = {}

    for pattern in missing_patterns:
        key = tuple(pattern)
        SSIM_dict[key] = [0.0]*len(opt.MODALITY_LIST)
        PSNR_dict[key] = [0.0]*len(opt.MODALITY_LIST)
        MAE_dict[key] = [0.0]*len(opt.MODALITY_LIST)
        count_dict[key] = [0]*len(opt.MODALITY_LIST)

    fsq_levels = grad_model.fsq_levels
    with torch.no_grad():
        for test_step, sample in enumerate(tqdm(test_loader)):
            volumes = sample['images'].to(device)  # shape: (B, C, H, W)
            # check_is_legal(volumes)
            b, c_modal, h, w = volumes.size()

            for pattern in missing_patterns:
                key = tuple(pattern)

                # Create partial volume by zeroing out missing channels
                masked_volumes = masking_img(volumes, pattern)
                out_levels = grad_model(masked_volumes)

                codes_quant = []
                for j, o in enumerate(out_levels):
                    prob_o = torch.sigmoid(o)
                    pred_index = (prob_o > 0.5).sum(dim=1)
                    codes_quant.append(scale_to_codes(pred_index, fsq_levels[j]))
                codes_quant = torch.stack(codes_quant, dim=1)

                encoded_priors = rec_model.prior_encoder(masked_volumes)
                output_volumes = rec_model.decoder(codes_quant, encoded_priors)
                
                for ch in range(c_modal):
                    if pattern[ch] == 0:  # Only evaluate missing modalities
                        gt_slice = volumes[:, ch, :, :].unsqueeze(1)  # (B,1,H,W)
                        output_slice = output_volumes[:, ch, :, :].unsqueeze(1)
                        output_slice = output_slice.clamp(0, 1)

                        # Compute metrics
                        gt_cpu = gt_slice.cpu()
                        out_cpu = output_slice.cpu()
                        psnr_val, ssim_val, mae_val = get_metrics(out_cpu, gt_cpu)

                        PSNR_dict[key][ch] += psnr_val
                        SSIM_dict[key][ch] += ssim_val
                        MAE_dict[key][ch] += mae_val
                        count_dict[key][ch] += 1

                        # Visualization (optional)
                        key_str = str(key).replace("(", "").replace(")", "").replace(", ", "_")
                        if opt.VISUALIZE and test_step == len(test_loader)//2:
                            concat_pred = show_img(gt_cpu, out_cpu)
                            cv2.imshow(f'Pattern_{key_str}_Predictions_{opt.MODALITY_LIST[ch]}', concat_pred)
                            cv2.waitKey(0)

                        # Save prediction images (optional)
                        if opt.TEST_SAVE:
                            vol_cpu = volumes.cpu().numpy()
                            preds_cpu = out_cpu.numpy()
                            cv2.imwrite(f"{result_path}/{test_step}_pattern_{key_str}_gt_{opt.MODALITY_LIST[ch]}.png", vol_cpu[0, ch]*255)
                            cv2.imwrite(f"{result_path}/{test_step}_pattern_{key_str}_pred_{opt.MODALITY_LIST[ch]}_PSNR_{psnr_val:.3f}.png", preds_cpu[0, 0]*255)

    # Compute per-pattern and per-channel means
    final_metrics = {}
    for pattern in missing_patterns:
        key = tuple(pattern)
        c_psnr = []
        c_ssim = []
        c_mae = []

        for ch in range(len(opt.MODALITY_LIST)):
            n_ch = count_dict[key][ch]
            if n_ch > 0:  # Only consider reconstructed modalities
                c_psnr.append(PSNR_dict[key][ch] / n_ch)
                c_ssim.append(SSIM_dict[key][ch] / n_ch)
                c_mae.append(MAE_dict[key][ch] / n_ch)
            else:
                # No reconstruction for this channel in this pattern
                c_psnr.append(None)
                c_ssim.append(None)
                c_mae.append(None)

        valid_psnr = [v for v in c_psnr if v is not None]
        valid_ssim = [v for v in c_ssim if v is not None]
        valid_mae = [v for v in c_mae if v is not None]

        mean_psnr = np.mean(valid_psnr) if valid_psnr else 0
        mean_ssim = np.mean(valid_ssim) if valid_ssim else 0
        mean_mae = np.mean(valid_mae) if valid_mae else 0
        
        var_psnr = np.var(valid_psnr) if len(valid_psnr) > 1 else 0
        var_ssim = np.var(valid_ssim) if len(valid_ssim) > 1 else 0
        var_mae = np.var(valid_mae) if len(valid_mae) > 1 else 0

        final_metrics[key] = {
            "PSNR_list": c_psnr,
            "SSIM_list": c_ssim,
            "MAE_list": c_mae,
            "Mean_PSNR": mean_psnr,
            "Mean_SSIM": mean_ssim,
            "Mean_MAE": mean_mae,
            "Var_PSNR": var_psnr,
            "Var_SSIM": var_ssim,
            "Var_MAE": var_mae,
        }
    
    # Collect all individual values for variance calculation
    all_ssim_values = []
    all_psnr_values = []
    all_mae_values = []
    # Separate patterns by number of inputs (1 vs multiple)
    single_input_ssim = []
    single_input_psnr = []
    single_input_mae = []

    multiple_input_ssim = []
    multiple_input_psnr = []
    multiple_input_mae = []

    for pattern in missing_patterns:
        key = tuple(pattern)
        num_inputs = sum(pattern)  # Count number of available inputs (1s)
        for ch in range(len(opt.MODALITY_LIST)):
            if pattern[ch] == 0:  # Only consider missing modalities
                n_ch = count_dict[key][ch]
                if n_ch > 0:
                    ssim_val = SSIM_dict[key][ch] / n_ch
                    psnr_val = PSNR_dict[key][ch] / n_ch
                    mae_val = MAE_dict[key][ch] / n_ch
                    
                    if num_inputs == 1:
                        single_input_ssim.append(ssim_val)
                        single_input_psnr.append(psnr_val)
                        single_input_mae.append(mae_val)
                    else:
                        multiple_input_ssim.append(ssim_val)
                        multiple_input_psnr.append(psnr_val)
                        multiple_input_mae.append(mae_val)

                    # Collect individual average values for each pattern/channel combination
                    all_ssim_values.append(ssim_val)
                    all_psnr_values.append(psnr_val)
                    all_mae_values.append(mae_val)

    overall_avg_ssim = np.mean(all_ssim_values) if len(all_ssim_values) > 0 else 0
    overall_avg_psnr = np.mean(all_psnr_values) if len(all_psnr_values) > 0 else 0
    overall_avg_mae = np.mean(all_mae_values) if len(all_mae_values) > 0 else 0
    overall_var_ssim = np.var(all_ssim_values) if len(all_ssim_values) > 0 else 0
    overall_var_psnr = np.var(all_psnr_values) if len(all_psnr_values) > 0 else 0
    overall_var_mae = np.var(all_mae_values) if len(all_mae_values) > 0 else 0

    # Compute statistics for single input patterns
    single_avg_ssim = np.mean(single_input_ssim) if len(single_input_ssim) > 0 else 0
    single_avg_psnr = np.mean(single_input_psnr) if len(single_input_psnr) > 0 else 0
    single_avg_mae = np.mean(single_input_mae) if len(single_input_mae) > 0 else 0
    single_var_ssim = np.var(single_input_ssim) if len(single_input_ssim) > 1 else 0
    single_var_psnr = np.var(single_input_psnr) if len(single_input_psnr) > 1 else 0
    single_var_mae = np.var(single_input_mae) if len(single_input_mae) > 1 else 0

    # Compute statistics for multiple input patterns
    multiple_avg_ssim = np.mean(multiple_input_ssim) if len(multiple_input_ssim) > 0 else 0
    multiple_avg_psnr = np.mean(multiple_input_psnr) if len(multiple_input_psnr) > 0 else 0
    multiple_avg_mae = np.mean(multiple_input_mae) if len(multiple_input_mae) > 0 else 0
    multiple_var_ssim = np.var(multiple_input_ssim) if len(multiple_input_ssim) > 1 else 0
    multiple_var_psnr = np.var(multiple_input_psnr) if len(multiple_input_psnr) > 1 else 0
    multiple_var_mae = np.var(multiple_input_mae) if len(multiple_input_mae) > 1 else 0

    # Print results
    print("\n===== Evaluation Results (Per Pattern) =====")
    for pattern_key, metrics in final_metrics.items():
        pattern_str = ",".join(str(v) for v in pattern_key)
        print(f"Pattern [{pattern_str}]")
        print(f"  PSNR_list: {metrics['PSNR_list']}")
        print(f"  SSIM_list: {metrics['SSIM_list']}")
        print(f"  MAE_list : {metrics['MAE_list']}")
        print(f"  Mean_PSNR: {metrics['Mean_PSNR']:.4f} (Var: {metrics['Var_PSNR']:.4f})")
        print(f"  Mean_SSIM: {metrics['Mean_SSIM']:.4f} (Var: {metrics['Var_SSIM']:.4f})")
        print(f"  Mean_MAE : {metrics['Mean_MAE']:.4f} (Var: {metrics['Var_MAE']:.4f})\n")

    print("===== Statistics by Input Number =====")
    print(f"Single Input (1 input):")
    print(f"  PSNR: {single_avg_psnr:.4f} (Var: {single_var_psnr:.4f})")
    print(f"  SSIM: {single_avg_ssim:.4f} (Var: {single_var_ssim:.4f})")
    print(f"  MAE : {single_avg_mae:.4f} (Var: {single_var_mae:.4f})\n")
    
    print(f"Multiple Inputs (>1 inputs):")
    print(f"  PSNR: {multiple_avg_psnr:.4f} (Var: {multiple_var_psnr:.4f})")
    print(f"  SSIM: {multiple_avg_ssim:.4f} (Var: {multiple_var_ssim:.4f})")
    print(f"  MAE : {multiple_avg_mae:.4f} (Var: {multiple_var_mae:.4f})\n")

    print("===== Overall Average Across All Patterns & Missing Channels =====")
    print(f"  Overall_PSNR: {overall_avg_psnr:.4f} (Var: {overall_var_psnr:.4f})")
    print(f"  Overall_SSIM: {overall_avg_ssim:.4f} (Var: {overall_var_ssim:.4f})")
    print(f"  Overall_MAE : {overall_avg_mae:.4f} (Var: {overall_var_mae:.4f})\n")

    save_txt_file = os.path.join(result_path, "../evaluation_results.txt")
    with open(save_txt_file, 'w') as f:
        f.write("===== Evaluation Results (Per Pattern) =====\n")
        for pattern_key, metrics in final_metrics.items():
            pattern_str = ",".join(str(v) for v in pattern_key)
            f.write(f"Pattern [{pattern_str}]\n")
            f.write(f"  PSNR_list: {metrics['PSNR_list']}\n")
            f.write(f"  SSIM_list: {metrics['SSIM_list']}\n")
            f.write(f"  MAE_list : {metrics['MAE_list']}\n")
            f.write(f"  Mean_PSNR: {metrics['Mean_PSNR']:.4f} (Var: {metrics['Var_PSNR']:.4f})\n")
            f.write(f"  Mean_SSIM: {metrics['Mean_SSIM']:.4f} (Var: {metrics['Var_SSIM']:.4f})\n")
            f.write(f"  Mean_MAE : {metrics['Mean_MAE']:.4f} (Var: {metrics['Var_MAE']:.4f})\n\n")

        f.write("===== Statistics by Input Number =====\n")
        f.write(f"Single Input (1 input):\n")
        f.write(f"  PSNR: {single_avg_psnr:.4f} (Var: {single_var_psnr:.4f})\n")
        f.write(f"  SSIM: {single_avg_ssim:.4f} (Var: {single_var_ssim:.4f})\n")
        f.write(f"  MAE : {single_avg_mae:.4f} (Var: {single_var_mae:.4f})\n\n")
        
        f.write(f"Multiple Inputs (>1 inputs):\n")
        f.write(f"  PSNR: {multiple_avg_psnr:.4f} (Var: {multiple_var_psnr:.4f})\n")
        f.write(f"  SSIM: {multiple_avg_ssim:.4f} (Var: {multiple_var_ssim:.4f})\n")
        f.write(f"  MAE : {multiple_avg_mae:.4f} (Var: {multiple_var_mae:.4f})\n\n")

        f.write("===== Overall Average Across All Patterns & Missing Channels =====\n")
        f.write(f"  Overall_PSNR: {overall_avg_psnr:.4f} (Var: {overall_var_psnr:.4f})\n")
        f.write(f"  Overall_SSIM: {overall_avg_ssim:.4f} (Var: {overall_var_ssim:.4f})\n")
        f.write(f"  Overall_MAE : {overall_avg_mae:.4f} (Var: {overall_var_mae:.4f})\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/params_ixi.yaml', help='config path (*.yaml)')
    parser.add_argument('--save_path', type=str, help='save path', default='../models/cb_grad_IXI_20260319-193259')
    parser.add_argument('--grad_name', type=str, default='cb_grad')

    parser.add_argument('--naf_dim', type=int, default=32, help='NAF dimension (default: 32)')
    parser.add_argument('--naf_depth', type=int, default=6, help='NAF depth (default: 6), totalling 27 blocks')
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8,8,8,5,5,5])

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    opt = Config(config_path=args.config)

    rand_seed(opt.RANDOM_SEED)
    
    evaluate_records = []
    db_test = BaseDataSets(
        base_dir=opt.DATA_PATH, 
        dataset=opt.DATASET,
        modality_list = opt.MODALITY_LIST,
        split='test'
    )
    test_loader = DataLoader(db_test, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
    rec_model = CodeBrain(args, opt).to(device)
    grad_model = CodeBrain_Grader(args, opt).to(device)

    ckpt = torch.load(os.path.join(args.save_path, '{}_{}_best.pth'.format(args.grad_name, opt.DATASET)), weights_only=True, map_location=device)
    
    rec_model.load_state_dict(ckpt['rec_model'])
    rec_model.to(device)
    rec_model.eval()

    grad_model.load_state_dict(ckpt['grad_model'])
    grad_model.to(device)
    grad_model.eval()

    result_path = os.path.join(args.save_path, 'results')
    os.makedirs(result_path, exist_ok=True)
    evaluate_grad(rec_model, grad_model, test_loader, opt, result_path, device)