#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_base_training.py 8
#srun49 python3 -m tools.detection.misc.checkpoint_surgery --src1 ./work_dirs/tfa_faster_rcnn_r101_fpn_voc_split3_base_training/latest.pth --method randinit --save-dir work_dirs/tfa_faster_rcnn_r101_fpn_voc_split3_base_training
#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_1shot_ft.py 8
#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_2shot_ft.py 8
#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_3shot_ft.py 8
#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_5shot_ft.py 8
#srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_faster_rcnn_r101_fpn_voc_split3_10shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_1shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_2shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_3shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_5shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_10shot_ft.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_3shot_ft_cl.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_5shot_ft_cl.py 8
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_faster_rcnn_r101_fpn_voc_split3_10shot_ft_cl.py 8
