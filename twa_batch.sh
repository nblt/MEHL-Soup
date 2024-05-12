#!/bin/bash

lr=0.01
wd=0.
layer=0 
devices=0
port=1234

model=clip-vit-b32
dataset=ImageNet
DST=../models/

batch=18
CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.launch --nproc_per_node 1 --master_port $port train_twa_layer_batch_dir.py \
    --lr $lr --batch-size 128 --wd $wd --epochs 5 \
    --cosine-lr --optimizer adamw --layer-size $layer \
    --model-location $DST \
    --datasets $dataset --params_start 0 --params_end  72 --arch $model --val_ratio $val_ratio \
    --optimizer adamw --split --ddp --randomseed $seed \
    --finetune --model_batch $batch --models_epochs 1
