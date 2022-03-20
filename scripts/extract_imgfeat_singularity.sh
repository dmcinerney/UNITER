#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=1
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --job-name=uniter
#module load discovery/2019-02-21
module load singularity
#IMG_DIR=$1
#OUT_DIR=$2
PATH_TO_STORAGE=/scratch/mcinerney.de/uniter_data
IMG_DIR=$PATH_TO_STORAGE/imagenome/normal_jpgs/val
OUT_DIR=$PATH_TO_STORAGE/imagenome/normal_fastrcnn_features/val

set -e

echo "extracting image features..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

singularity exec --nv --ipc \
    --bind $IMG_DIR:/img \
    --bind $OUT_DIR:/output \
    uniter_features_img \
    bash -c "cd /src; python tools/generate_npz.py --gpu 0"

echo "done"
