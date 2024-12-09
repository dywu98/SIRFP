#!/bin/bash/



GPU_NUM=4
NUM_WORKERS=4


DATASET='CS'
BALANCE=2
DATA_PARA='{"resample":true}'
#MODEL_NAME='deeplabv3'
# MODEL_NAME='deeplabv3p'
MODEL_NAME='deeplabv3p'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"os":8,"mg_unit":[1,2,4],"inplanes":128}'
# INPUT_SIZE='1024,2048'
INPUT_SIZE='769,769'
ALIGN_CORNER='True'
BS=8
TEST_BS=${GPU_NUM}
LOSS_TYPE='gsrl'
LOSS_PARA='{"ds_weight":0.4}'
OPTIM='sgd'
LEARNING_RATE=0.01
WEIGHT_DECAY=0.001
WARMUP=1000
NUM_STEPS=36000
SAVE_PRED_EVERY=800
SAVE_STEPS=$(($NUM_STEPS - `expr 7 \* ${SAVE_PRED_EVERY}`))

# SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_p08_finetune_iter2
# RESUME=ckpt/CS/graph_edge_multi_p08_finetune_iter1_deeplabv3p/simi_prune_08/pruned.pth
# CHANNEL_CFG=ckpt/CS/graph_edge_multi_p08_finetune_iter1_deeplabv3p/simi_prune_08/channel_cfg.pth

SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_0711resume_finetune_iter2
CHANNEL_CFG=ckpt/CS/graph_edge_multi_0711resume_finetune_iter2_deeplabv3p/channel_cfg.pth

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29888}

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=$MASTER_ADDR --master_port=$PORT train_simi_flex.py --dataset ${DATASET} --balance ${BALANCE} --data-para ${DATA_PARA} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --loss-type ${LOSS_TYPE} --loss-para ${LOSS_PARA} --random-mirror --random-brightness --random-scale --optim ${OPTIM} --learning-rate ${LEARNING_RATE} --warmup ${WARMUP} --weight-decay ${WEIGHT_DECAY} --num-workers ${NUM_WORKERS} --num-steps ${NUM_STEPS} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --batch-size ${BS} --random-seed 42 --snapshot-dir ${SNAPSHOT_DIR}_${MODEL_NAME} --save-pred-every ${SAVE_PRED_EVERY} --save-steps ${SAVE_STEPS} --resume ${RESUME} --channel-cfg ${CHANNEL_CFG} --prune-type ${PRUNE_TYPE} --prune-method ${PRUNE_METHOD} 

#ss test
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_withflops_acc.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}

# python evaluate_withflops_acc.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}

# python totrt.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --align-corner ${ALIGN_CORNER} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --channel-cfg ${CHANNEL_CFG} --save-dir ${SNAPSHOT_DIR}_${MODEL_NAME}

#ms test
# WHOLE='True'; MS='0.5,0.75,1,1.25,1.5,1.75'; FLIP='True'
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_withflops_acc.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}
MODEL_NAME='deeplabv3'
SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_0711resume_finetune_iter2
CHANNEL_CFG=ckpt/CS/graph_edge_multi_0711resume_finetune_iter2_deeplabv3/channel_cfg.pth

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_withflops_acc.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}


# python totrt.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --align-corner ${ALIGN_CORNER} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --channel-cfg ${CHANNEL_CFG} --save-dir ${SNAPSHOT_DIR}_${MODEL_NAME}



SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_p07_finetune_iter2
CHANNEL_CFG=ckpt/CS/graph_edge_multi_p07_finetune_iter2_deeplabv3/channel_cfg.pth

python totrt.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --align-corner ${ALIGN_CORNER} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --channel-cfg ${CHANNEL_CFG} --save-dir ${SNAPSHOT_DIR}_${MODEL_NAME}


SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_p08_finetune_iter2
CHANNEL_CFG=ckpt/CS/graph_edge_multi_p08_finetune_iter2_deeplabv3/channel_cfg.pth

python totrt.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --align-corner ${ALIGN_CORNER} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --channel-cfg ${CHANNEL_CFG} --save-dir ${SNAPSHOT_DIR}_${MODEL_NAME}
