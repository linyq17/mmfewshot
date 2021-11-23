#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_base_training.py 8
#srun38 python3 -m tools.detection.misc.checkpoint_surgery --src1 ./work_dirs/tfa_faster_rcnn_r101_fpn_voc_split2_base_training/latest.pth --method randinit --save-dir work_dirs/tfa_faster_rcnn_r101_fpn_voc_split2_base_training
#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_1shot_ft.py 8
#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_2shot_ft.py 8
#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_3shot_ft.py 8
#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_5shot_ft.py 8
#srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_faster_rcnn_r101_fpn_voc_split2_10shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_1shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_2shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_3shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_5shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_10shot_ft.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_3shot_ft_cl.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_5shot_ft_cl.py 8
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_faster_rcnn_r101_fpn_voc_split2_10shot_ft_cl.py 8
srun38 bash ./tools/detection/dist_test.sh configs/detection/fsce/voc/split1/tfa_faster_rcnn_r101_fpn_voc_split1_1shot_ft work_dirs/tfa_faster_rcnn_r101_fpn_voc_split1_1shot_ft/latest.pth 8 --eval mAP
