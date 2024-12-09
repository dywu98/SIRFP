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
BS=32
TEST_BS=${GPU_NUM}
LOSS_TYPE='ce'
LOSS_PARA='{"ds_weight":0.4}'
OPTIM='sgd'
LEARNING_RATE=0.01
WEIGHT_DECAY=0.0001
PRUNE_TYPE='simi'
WARMUP=-1
NUM_STEPS=16000
SAVE_PRED_EVERY=5000
SAVE_STEPS=$(($NUM_STEPS - `expr 7 \* ${SAVE_PRED_EVERY}`))
SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_pretrain
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-39888}
PRUNE_METHOD='graph_edge_pruning'

# #train
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train_simi_flex.py --dataset ${DATASET} --balance ${BALANCE} --data-para ${DATA_PARA} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --loss-type ${LOSS_TYPE} --loss-para ${LOSS_PARA} --random-mirror --random-brightness --random-scale --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --optim ${OPTIM} --learning-rate ${LEARNING_RATE} --warmup ${WARMUP} --weight-decay ${WEIGHT_DECAY} --num-workers ${NUM_WORKERS} --num-steps ${NUM_STEPS} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --batch-size ${BS} --random-seed 42 --snapshot-dir ${SNAPSHOT_DIR}_${MODEL_NAME} --save-pred-every ${SAVE_PRED_EVERY} --save-steps ${SAVE_STEPS} --prune-type ${PRUNE_TYPE} --prune-method ${PRUNE_METHOD} 

# #ss test
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False'


DATASET='ADE'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"mg_unit":[1,1,1]}'
ALIGN_CORNER='False'
PRUNE_RATIO=0.3
SNAPSHOT_DIR=ckpt/ADE/graph_edge_multi_pretrain_deeplabv3/graph_edge_multi_prune_06
RESUME_DIR=ckpt/ADE/graph_edge_multi_pretrain_deeplabv3/ADE_scenes_16000.pth
SCORE_DIR=ckpt/ADE/graph_edge_multi_pretrain_deeplabv3/score.pth


python prune_graph_edge.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --prune-ratio ${PRUNE_RATIO} --save-path ${SNAPSHOT_DIR} --model-path ${RESUME_DIR} --score-path ${SCORE_DIR} 

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_withflops.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}/pruned.pth --save-predict 'False' --channel-cfg ${SNAPSHOT_DIR}/channel_cfg.pth


# #######################################################
# # iter1

GPU_NUM=4
NUM_WORKERS=4

DATASET='ADE'
BALANCE=2
DATA_PARA='{"resample":true}'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"mg_unit":[1,1,1]}'
INPUT_SIZE='512,512'
ALIGN_CORNER='False'
LONG_SIZE=-1
SHORT_SIZE=512
BS=32
TEST_BS=${GPU_NUM}
LOSS_TYPE='gsrl'
LOSS_PARA='{"ds_weight":0.4}'
OPTIM='sgd'
LEARNING_RATE=0.01
WEIGHT_DECAY=0.0001
PRUNE_TYPE='simi'
WARMUP=-1
NUM_STEPS=16000
SAVE_PRED_EVERY=5000
SAVE_STEPS=$(($NUM_STEPS - `expr 7 \* ${SAVE_PRED_EVERY}`))
SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_finetune_iter1
RESUME=ckpt/ADE/graph_edge_multi_pretrain_deeplabv3/graph_edge_multi_prune_06/pruned.pth
CHANNEL_CFG=ckpt/ADE/graph_edge_multi_pretrain_deeplabv3/graph_edge_multi_prune_06/channel_cfg.pth
PRUNE_METHOD='graph_edge_pruning'

# # #train
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train_simi_flex.py --dataset ${DATASET} --balance ${BALANCE} --data-para ${DATA_PARA} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --loss-type ${LOSS_TYPE} --loss-para ${LOSS_PARA} --random-mirror --random-brightness --random-scale --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --optim ${OPTIM} --learning-rate ${LEARNING_RATE} --warmup ${WARMUP} --weight-decay ${WEIGHT_DECAY} --num-workers ${NUM_WORKERS} --num-steps ${NUM_STEPS} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --batch-size ${BS} --random-seed 42 --snapshot-dir ${SNAPSHOT_DIR}_${MODEL_NAME} --save-pred-every ${SAVE_PRED_EVERY} --save-steps ${SAVE_STEPS} --resume ${RESUME} --channel-cfg ${CHANNEL_CFG} --prune-type ${PRUNE_TYPE} --prune-method ${PRUNE_METHOD} 


# #ss test
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}



DATASET='ADE'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"mg_unit":[1,1,1]}'
ALIGN_CORNER='False'
PRUNE_RATIO=0.595
SNAPSHOT_DIR=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/graph_edge_multi_prune_06
RESUME_DIR=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/ADE_scenes_16000.pth
SCORE_DIR=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/score.pth
CHANNEL_CFG=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/channel_cfg.pth

python prune_graph_edge_multi.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --prune-ratio ${PRUNE_RATIO} --save-path ${SNAPSHOT_DIR} --model-path ${RESUME_DIR} --score-path ${SCORE_DIR} --channelcfg-path ${CHANNEL_CFG}

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}/pruned.pth --save-predict 'False' --channel-cfg ${SNAPSHOT_DIR}/channel_cfg.pth






GPU_NUM=4
NUM_WORKERS=4

DATASET='ADE'
BALANCE=2
DATA_PARA='{"resample":true}'
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
LOSS_TYPE='gsrl'
LOSS_PARA='{"ds_weight":0.4}'
OPTIM='sgd'
LEARNING_RATE=0.01
WEIGHT_DECAY=0.0005
WARMUP=2000
NUM_STEPS=160000
SAVE_PRED_EVERY=5000
SAVE_STEPS=$(($NUM_STEPS - `expr 7 \* ${SAVE_PRED_EVERY}`))
SNAPSHOT_DIR=ckpt/${DATASET}/graph_edge_multi_finetune_iter2
RESUME=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/graph_edge_multi_prune_06/pruned.pth
CHANNEL_CFG=ckpt/ADE/graph_edge_multi_finetune_iter1_deeplabv3/graph_edge_multi_prune_06/channel_cfg.pth


# #train
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train_simi_flex.py --dataset ${DATASET} --balance ${BALANCE} --data-para ${DATA_PARA} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --loss-type ${LOSS_TYPE} --loss-para ${LOSS_PARA} --random-mirror --random-brightness --random-scale --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --optim ${OPTIM} --learning-rate ${LEARNING_RATE} --warmup ${WARMUP} --weight-decay ${WEIGHT_DECAY} --num-workers ${NUM_WORKERS} --num-steps ${NUM_STEPS} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --batch-size ${BS} --random-seed 42 --snapshot-dir ${SNAPSHOT_DIR}_${MODEL_NAME} --save-pred-every ${SAVE_PRED_EVERY} --save-steps ${SAVE_STEPS} --resume ${RESUME} --channel-cfg ${CHANNEL_CFG}


#ss test
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}

#ms test
WHOLE='True'; MS='0.5,0.75,1,1.25,1.5,1.75'; FLIP='True'
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} evaluate_withflops.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --batch-size ${TEST_BS} --shortsize ${SHORT_SIZE} --longsize ${LONG_SIZE} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ${SNAPSHOT_DIR}_${MODEL_NAME}/${DATASET}_scenes_${NUM_STEPS}.pth --save-predict 'False' --channel-cfg ${CHANNEL_CFG}

