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
#IMG_NPY=$1
#OUT_DIR=$2
PATH_TO_STORAGE=/scratch/mcinerney.de/uniter_data
IMG_NPY=$PATH_TO_STORAGE/imagenome/normal_npz/val
OUT_DIR=$PATH_TO_STORAGE/img_db/imagenome_normal

set -e

echo "converting image features ..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
NAME=$(basename $IMG_NPY)
singularity exec --ipc \
    --bind $(pwd):/src \
    --bind $OUT_DIR:/img_db \
    --bind $IMG_NPY:/$NAME \
    uniter_img \
    python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db

echo "done"
