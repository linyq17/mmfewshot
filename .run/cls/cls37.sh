#!/usr/bin/env bash
METHOD=$1
DATASET=$2
BACKBONE=$3
PORT=${4:-29500}

if [ $BACKBONE != 'wrn28x10' ]
then
  if [ $METHOD != 'neg_margin' ]
  then
    srun37g1 python ./tools/classification/train.py configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot.py
    srun37g1 python ./tools/classification/test.py configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot.py ./work_dirs/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
    srun37g1 python ./tools/classification/test.py configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_5shot.py ./work_dirs/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
  else
    srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot.py
    srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot.py ./work_dirs/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
    srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_5shot.py ./work_dirs/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
  fi
else
  if [ $METHOD != 'neg_margin' ]
  then
    srun37g2 bash ./tools/classification/dist_train.sh configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot_2gpu.py 2 ${PORT}
    srun37g1 python ./tools/classification/test.py configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot.py ./work_dirs/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot_2gpu/best_accuracy_mean.pth
    srun37g1 python ./tools/classification/test.py configs/classification/${METHOD}/${DATASET}/${METHOD}_${BACKBONE}_${DATASET}_5way_5shot.py ./work_dirs/${METHOD}_${BACKBONE}_${DATASET}_5way_1shot_2gpu/best_accuracy_mean.pth
  else
    srun37g2 bash ./tools/classification/dist_train.sh configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot_2gpu.py 2 ${PORT}
    srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot.py ./work_dirs/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
    srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/${DATASET}/neg_cosine_${BACKBONE}_${DATASET}_5way_5shot.py ./work_dirs/neg_cosine_${BACKBONE}_${DATASET}_5way_1shot/best_accuracy_mean.pth
  fi
fi
