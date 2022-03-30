# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=/home/jered/Documents/data/uniter_data/txt_db/
IMG_DIR=/home/jered/Documents/data/uniter_data/img_db/
OUTPUT=/home/jered/Documents/data/uniter_data/finetune
PRETRAIN_DIR=/home/jered/Documents/data/uniter_data/pretrained
#TXT_DB=$1
#IMG_DIR=$2
#OUTPUT=$3
#PRETRAIN_DIR=$4

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    --mount src=/home/jered/Documents/data/uniter_data/imagenome,dst=/imagenome,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p 8888:8888 \
    -w /src chenrocks/uniter


#    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \