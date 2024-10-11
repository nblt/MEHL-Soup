#!/bin/bash

lr=0.01
wd=0.
devices=0
port=1234
model=clip-vit-b32
dataset=ImageNet
model_dir=../models/
batch=18
CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port $port MEHL_soup_layer.py \
    --lr $lr --batch-size 128 --wd $wd --epochs 5 \
    --cosine-lr --optimizer adamw --model-location $model_dir \
    --datasets $dataset --params_start 0 --params_end  72 --arch $model \
    --optimizer adamw --split --ddp --randomseed $seed \
    --finetune --model_batch $batch --models_epochs 1