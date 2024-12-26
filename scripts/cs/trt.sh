#!/bin/bash/


DATASET='CS'
INPUT_SIZE='1025,2049'
MODEL_NAME='deeplabv3'
MODEL_PARA='{}'
BACKBONE='resnet50'
BACKBONE_PARA='{"os":8,"mg_unit":[1,2,4],"inplanes":128}'
ALIGN_CORNER='True'
SAVE_DIR=ckpt/CS/your_path/trt.pth
RESUME=ckpt/CS/your_path/your_pth.pth
CHANNEL_CFG=ckpt/CS/your_path/channel_cfg.pth


python totrt.py --dataset ${DATASET} --input-size ${INPUT_SIZE} --model ${MODEL_NAME} --model-para ${MODEL_PARA} --backbone ${BACKBONE} --backbone-para ${BACKBONE_PARA} --align-corner ${ALIGN_CORNER} --restore-from ${RESUME} --channel-cfg ${CHANNEL_CFG} --save-dir ${SAVE_DIR} 


#ss test
python -m torch.distributed.launch --nproc_per_node=1 evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --backbone ${BACKBONE} --batch-size 1 --whole 'True' --flip 'False' --input-size ${INPUT_SIZE} --align-corner ${ALIGN_CORNER} --ms '1' --num-workers 1 --restore-from ${SAVE_DIR} --save-predict 'False' --use-trt 'True'
