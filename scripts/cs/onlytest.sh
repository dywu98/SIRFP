#!/bin/bash/

GPU_NUM=4
NUM_WORKERS=4


DATASET='CS'
BALANCE=2
DATA_PARA='{"resample":true}'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"os":8,"mg_unit":[1,2,4],"inplanes":128}'
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
SNAPSHOT_DIR=ckpt/CS/dcfp_finetune
# RESUME=ckpt/CS/simi2_pretrain_0612_deeplabv3/simi_prune_06/pruned.pth
# CHANNEL_CFG=ckpt/CS/random_pretrain_deeplabv3/random_prune_06/channel_cfg.pth
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29888}

#ss test
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/random_pretrain_deeplabv3/random_prune_06/pruned.pth --save-predict 'False' --channel-cfg ckpt/CS/random_pretrain_deeplabv3/random_prune_06/channel_cfg.pth
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/dcfp_pretrain_deeplabv3/dcfp_prune_06/pruned.pth --save-predict 'False' --channel-cfg ckpt/CS/dcfp_pretrain_deeplabv3/dcfp_prune_06/channel_cfg.pth
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi_pretrain_deeplabv3/simi_prune_06/pruned.pth --save-predict 'False' --channel-cfg ckpt/CS/simi_pretrain_deeplabv3/simi_prune_06/channel_cfg.pth
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi2_pretrain_0612_deeplabv3/simi_prune_06/pruned.pth --save-predict 'False' --channel-cfg ckpt/CS/simi2_pretrain_0612_deeplabv3/simi_prune_06/channel_cfg.pth

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi2_pretrain_0612_deeplabv3/CS_scenes_4000.pth --save-predict 'False' 
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi3_pretrain_deeplabv3/CS_scenes_4000.pth --save-predict 'False' 
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi4_pretrain_deeplabv3/CS_scenes_4000.pth --save-predict 'False' 
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/simi4_pretrain_deeplabv3/CS_scenes_3200.pth --save-predict 'False' 






#ms test
WHOLE='True'; MS='0.5,0.75,1,1.25,1.5,1.75'; FLIP='True'
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/dcfp_finetune_deeplabv3/CS_scenes_36000.pth --save-predict 'True' --channel-cfg ckpt/CS/dcfp_finetune_deeplabv3/channel_cfg.pth

SNAPSHOT_DIR=ckpt/CS/graph_edge_multi_0711resume_finetune_iter2
# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/graph_edge_multi_0711resume_finetune_iter2_deeplabv3/CS_scenes_36000.pth --save-predict 'True' --channel-cfg ckpt/CS/graph_edge_multi_0711resume_finetune_iter2_deeplabv3/channel_cfg.pth

SNAPSHOT_DIR=ckpt/CS/graph_edge_wrong_finetune
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_amp.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole ${WHOLE} --flip ${FLIP} --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms ${MS} --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/graph_edge_wrong_finetune_deeplabv3/CS_scenes_36000.pth --save-predict 'True' 
