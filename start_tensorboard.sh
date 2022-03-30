module load anaconda3/3.7
module load discovery/2019-02-21
source activate tnsrbrd
ssh login-00 -f -N -T -R 6006:localhost:6006
tensorboard --logdir /scratch/mcinerney.de/uniter_data/output/experiment_1gpu/log
