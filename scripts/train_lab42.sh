

echo `pwd`
echo "PYTHON: `which python`"
echo "WANDB: `which wandb`"
echo "SWEEP: $SWEEP, $SLURM_JOB_ID"

echo 'Starting'
PYTHON='/home/rob/environments/pyt310/bin/python3'
${PYTHON} imagenet/imagenet_train.py \
    --base_learning_rate=0.1 \
    --disable_dp \
    --eval_every_n_steps=500 \
    --finetune_cut_last_layer=True \
    --finetune_path=finetuned_models/places365_resnet18_20220314.npz \
    --logit_clip=tanh \
    --max_eval_batches=5 \
    --model=resnet18  \
    --model_dir=saved_models/baselearningrate__$RANDOM \
    --num_train_epochs=25 \
    --tfds_data_dir="/home/rob/Documents/datasets/imagenet" \
    --train_device_batch_size=256
