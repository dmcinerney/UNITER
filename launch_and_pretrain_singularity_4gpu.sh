#!/bin/bash
#SBATCH -p reservation
#SBATCH --reservation=multigpu_test_mcinerney.de
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --export=1
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --job-name=uniter
#module load discovery/2019-02-21
module load singularity
#TXT_DB=$1
#IMG_DIR=$2
#OUTPUT=$3
#PRETRAIN_DIR=$4
PATH_TO_STORAGE=/scratch/mcinerney.de/uniter_data
TXT_DB=$PATH_TO_STORAGE/txt_db
IMG_DIR=$PATH_TO_STORAGE/img_db
OUTPUT=$PATH_TO_STORAGE/output
PRETRAIN_DIR=$PATH_TO_STORAGE/pretrained
singularity exec --nv --ipc \
    --bind $(pwd):/src \
    --bind $OUTPUT:/storage \
    --bind $PRETRAIN_DIR:/pretrain \
    --bind $TXT_DB:/txt \
    --bind $IMG_DIR:/img \
    uniter_img \
    horovodrun -np 4 -H localhost:4 python pretrain.py --config config/pretrain-imagenome-base-1gpu.json \
    --output_dir /storage/experiment_4gpu
