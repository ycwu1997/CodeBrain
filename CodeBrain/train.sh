cd code
nvidia-smi
# torchrun --nproc_per_node 1 --master_port 1247 train_rec.py --config ./configs/params_ixi.yaml --bs 8
torchrun --nproc_per_node 1 --master_port 1247 train_grad.py --config ./configs/params_ixi.yaml --rec_path ../models/cb_rec_IXI_20260318-112206 --bs 16