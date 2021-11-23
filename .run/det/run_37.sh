#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_base_training.py 8
#srun37 python3 -m tools.detection.misc.checkpoint_surgery --src1 ./work_dirs/tfa_faster_rcnn_r101_fpn_voc_split1_base_training/latest.pth --method randinit --save-dir work_dirs/tfa_faster_rcnn_r101_fpn_voc_split1_base_training
#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_1shot_finetuning.py 8
#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_2shot_finetuning.py 8
#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_3shot_finetuning.py 8
#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_5shot_finetuning.py 8
#srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc_split1_10shot_finetuning.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_1shot_ft.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_2shot_ft.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_3shot_ft.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_5shot_ft.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_10shot_ft.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_3shot_ft_cl.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_5shot_ft_cl.py 8
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_faster_rcnn_r101_fpn_voc_split1_10shot_ft_cl.py 8
