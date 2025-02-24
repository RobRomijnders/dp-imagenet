#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:A10:1
#SBATCH --job-name=betterdp
#SBATCH --constraint=gpunode

source "/var/scratch/${USER}/projects/betterdp/scripts/preamble.sh"

module load cuda12.1/toolkit/12.1

echo `pwd`
echo "PYTHON: `which python`"
echo "WANDB: `which wandb`"
echo "SWEEP: $SWEEP, $SLURM_JOB_ID"

MESSAGE="`date "+%Y-%m-%d__%H-%M-%S"` \t ${SLURM_JOB_ID} \t ${SLURM_JOB_NAME} \t ${SWEEP}  \t "
sed -i "1i$MESSAGE" "/var/scratch/${USER}/projects/betterdp/jobs.txt"

echo 'Starting'
PYTHON='/var/scratch/rromijnd/virtualenvs/pyt310/bin/python3'
${PYTHON} imagenet/imagenet_train.py \
    --base_learning_rate=0.1 \
    --disable_dp \
    --eval_every_n_steps=500 \
    --finetune_cut_last_layer=True \
    --finetune_path=finetuned_models/places365_resnet18_20220314.npz \
    --max_eval_batches=5 \
    --model=resnet18  \
    --model_dir=saved_models/baselearningrate__$RANDOM \
    --num_train_epochs=25 \
    --tfds_data_dir="/var/scratch/rromijnd/datasets/imagenet/" \
    --train_device_batch_size=256
