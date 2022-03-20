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
#ANN_DIR=$2
#OUT_DIR=$1
PATH_TO_STORAGE=/scratch/mcinerney.de/uniter_data
ANN_DIR=$PATH_TO_STORAGE/imagenome/normal_text
OUT_DIR=$PATH_TO_STORAGE/txt_db/imagenome_normal

set -e

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

#for SPLIT in 'test' 'val' 'train'; do
for SPLIT in 'val'; do
    echo "preprocessing ${SPLIT} annotations..."
    singularity exec --ipc \
        --bind $(pwd):/src \
        --bind $OUT_DIR:/txt_db \
        --bind $ANN_DIR:/ann \
        uniter_img \
        python prepro.py --annotation /ann/${SPLIT} \
                         --output /txt_db/${SPLIT}.db --task imagenome
done

echo "done"
