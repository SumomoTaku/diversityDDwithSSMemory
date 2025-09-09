export CUDA_VISIBLE_DEVICES=0

select_list="../misc/class_nette.txt"
all_list="../misc/class_indices.txt"

imagenet_dir="/home/user/Sumomo/Project/Dataset/ImageNet"

save_dir="../results"
#fine-tune the diffusion model
torchrun --nnode=1 --master_port=25678 ../main.py "train_dit" --distill \
  --dit_model "DiT-XL/2" --dit_ckpt "../pretrained_models/DiT-XL-2-256x256.pt" \
  --train_dir "${imagenet_dir}/train" --batch_size 8 --epochs 8 \
  --ckpt_freq 12000 --print_freq 1500 --save_dir ${save_dir} \
  --lr 1e-3

dit_path="${save_dir}/ckpt/0012000.pt"
# run sample generation
python ../main.py "sample" --dit_model "DiT-XL/2" --dit_ckpt "/home/user/Sumomo/Project/models/minimax.pt" \
  --save_dir ${save_dir} --select_list ${select_list} --all_cls  ${all_list} \
  --ipc 1

image_dir="${save_dir}/sample"
# run validation
python ../main.py "train_downstream" -d "imagenet" --train_dir ${image_dir} --val_dir "${imagenet_dir}/val"\
  -n resnet_ap --depth 18 --nclass 10 --norm_type instance  --slct_type random \
  --select_list ${select_list} --ipc 10 \
  --save_dir ${save_dir} --repeat 1 \
  --print_freq 200 --val_freq  20 \
