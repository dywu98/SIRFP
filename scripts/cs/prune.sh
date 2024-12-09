#!/bin/bash/


DATASET='CS'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"os":8,"mg_unit":[1,2,4],"inplanes":128}'
ALIGN_CORNER='True'
PRUNE_RATIO=0.7
SNAPSHOT_DIR=ckpt/CS/dcfp_pretrain_deeplabv3/dcfp_prune_07
RESUME_DIR=ckpt/CS/dcfp_pretrain_deeplabv3/CS_scenes_4000.pth
SCORE_DIR=ckpt/CS/dcfp_pretrain_deeplabv3/score.pth

GPU_NUM=4
NUM_WORKERS=4
NUM_STEPS=36000
INPUT_SIZE='769,769'
ALIGN_CORNER='True'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
TEST_BS=${GPU_NUM}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29888}

python prune.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --prune-ratio ${PRUNE_RATIO} --save-path ${SNAPSHOT_DIR} --model-path ${RESUME_DIR} --score-path ${SCORE_DIR} 

# python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$PORT evaluate_withflops_acc.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size ${TEST_BS} --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers ${NUM_WORKERS} --restore-from ckpt/CS/dcfp_pretrain_deeplabv3/dcfp_prune_07/pruned.pth --save-predict 'False' --channel-cfg ckpt/CS/dcfp_pretrain_deeplabv3/dcfp_prune_07/channel_cfg.pth


