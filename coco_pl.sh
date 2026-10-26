#!/bin/bash

dataset="COCO2014" #VOC2007, NUSWIDE

sample_partial_mode='global_random'
sample_partial_param='0.1'

data_root='./datasets/dataset'
save_dir="./checkpoint/${dataset}_pl/$sample_partial_param"
model_name='RN101'
image_size=448
two_trans=true
epochs=30
batch_size=32
works=4
display_interval=100
momentum=0.9
weight_decay=1e-4
t=0.6
model_ema=true
proj=false
early_stop=true

lr=1e-4
seed=1200

evaluate=false
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

python train_pl.py \
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
    --seed ${seed} \
    --dataset ${dataset} \
    --backbone ${model_name} \
    --save_dir ${save_dir} \
    --sample_partial_mode ${sample_partial_mode} \
    --sample_partial_param ${sample_partial_param} \
    ${extra_cmd}
