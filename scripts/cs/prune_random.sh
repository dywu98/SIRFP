#!/bin/bash/


DATASET='CS'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"os":8,"mg_unit":[1,2,4],"inplanes":128}'
ALIGN_CORNER='True'
PRUNE_RATIO=0.6
SNAPSHOT_DIR=ckpt/CS/random_pretrain_deeplabv3/random_prune_06
RESUME_DIR=ckpt/CS/random_pretrain_deeplabv3/CS_scenes_4000.pth
SCORE_DIR=ckpt/CS/random_pretrain_deeplabv3/score.pth

python prune_random.py --dataset ${DATASET} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --prune-ratio ${PRUNE_RATIO} --save-path ${SNAPSHOT_DIR} --model-path ${RESUME_DIR} --score-path ${SCORE_DIR} 



