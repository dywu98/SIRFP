#!/bin/bash/

GPU_NUM=4
NUM_WORKERS=4

DATASET='ADE'
BALANCE=0
DATA_PARA='{"resample":false}'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"mg_unit":[1,1,1]}'
INPUT_SIZE='512,512'
ALIGN_CORNER='False'
LONG_SIZE=-1
SHORT_SIZE=512
BS=16
TEST_BS=${GPU_NUM}
LOSS_TYPE='ce'
LOSS_PARA='{"ds_weight":0.4}'
OPTIM='sgd'
LEARNING_RATE=0.01
WEIGHT_DECAY=0.0001
PRUNE_TYPE='dcfp'
WARMUP=-1
NUM_STEPS=16000
SAVE_PRED_EVERY=5000
SAVE_STEPS=$(($NUM_STEPS - `expr 7 \* ${SAVE_PRED_EVERY}`))
SNAPSHOT_DIR=ckpt/${DATASET}/dcfp_pretrain

#train
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train.py --dataset ${DATASET} --balance ${BALANCE} --data-para ${DATA_PARA} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --loss-type ${LOSS_TYPE} --loss-para ${LOSS_PARA} --random-mirror --random-brightness --random-scale --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --optim ${OPTIM} --learning-rate ${LEARNING_RATE} --warmup ${WARMUP} --weight-decay ${WEIGHT_DECAY} --num-workers ${NUM_WORKERS} --num-steps ${NUM_STEPS} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --batch-size ${BS} --random-seed 42 --snapshot-dir ${SNAPSHOT_DIR}_${MODEL_NAME} --save-pred-every ${SAVE_PRED_EVERY} --save-steps ${SAVE_STEPS} --prune-type ${PRUNE_TYPE}

#ss test
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_withflops.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False'


