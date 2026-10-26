#!/bin/bash

dataset="MSCOCO"
setup='LargeLoss'
data_root='./datasets/dataset'
save_dir="./checkpoint/${dataset}_spl2/"
model_name='RN50'
image_size=224
two_trans=true
epochs=20
batch_size=128
works=4
display_interval=100
momentum=0.9
weight_decay=1e-4
t=0.6
lambda_c=0.01
T=0.8
model_ema=true
proj=true
early_stop=true

lr=5e-5
seed=1200

evaluate=false
resume="./checkpoint/${dataset}_spl/checkpoint_best.pth"
extra_cmd=""
if $two_trans
then
  extra_cmd="$extra_cmd --two_trans"
fi

if $model_ema
then
  extra_cmd="$extra_cmd --model_ema"
fi

if $proj
then
  extra_cmd="$extra_cmd --proj"
fi

if $evaluate
then
  extra_cmd="$extra_cmd --evaluate"
fi

if $early_stop
then
  extra_cmd="$extra_cmd --early_stop"
fi

# resume is not none
if [ -n "$resume" ];
then
  extra_cmd="$extra_cmd --resume $resume"
fi

python train_spl2.py \
    --data_root_dir ${data_root} \
    --image_size ${image_size} \
    --epochs ${epochs} \
    -b ${batch_size} \
    -j ${works} \
    --display_interval ${display_interval} \
    --lr ${lr} \
    --momentum ${momentum} \
    --w_d ${weight_decay} \
    --t ${t} \
    --lambda_c ${lambda_c} \
    --T ${T} \
    --seed ${seed} \
    --dataset ${dataset} \
    --setup ${setup} \
    --backbone ${model_name} \
    --save_dir ${save_dir} \
    ${extra_cmd}
