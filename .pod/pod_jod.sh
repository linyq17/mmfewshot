CONFIG=$1
GPUS=$2
WORKDIR=$3

cd /code/mmfewshot;
ln -sf /code/.cache /root;
ln -sf /data/detection/coco coco;
ln -sf /data/detection/voc/VOCdevkit VOCdevkit;
export PYTHONPATH=$(pwd):$PYTHONPATH;
bash ./tools/detection/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORKDIR} ${@:4}
