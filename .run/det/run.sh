### TFA VOC ###

srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/tfa_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth" &

srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py 8  &
srun48 python3 -m tools.detection.misc.initialize_bbox_head --src1 ./work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
  --method random_init --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training  &
srun38 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py 8 --work-dir ./work_dirs/tfa_r101_fpn_voc-split1_1shot-fine-tuning_infinte &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning.py 8 &

srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.py 8  &
srun49 python3 -m tools.detection.misc.initialize_bbox_head --src1 ./work_dirs/tfa_r101_fpn_voc-split2_base-training/latest.pth \
  --method random_init --save-dir work_dirs/tfa_r101_fpn_voc-split2_base-training  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_1shot-fine-tuning.py 8  --options "load_from=https://download.openmmlab.com/mmfewshot/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training_20211031_114820_random-init-bbox-head-3d4c632c.pth" &

srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning.py 8 &


srun37 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.py 8 &
srun48 python3 -m tools.detection.misc.initialize_bbox_head --src1 ./released_ckpt/tfa_r101_fpn_voc-split3_base-training/latest.pth \
  --method random_init --save-dir work_dirs/tfa_r101_fpn_voc-split3_base-training
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning.py 8 &


srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_10shot-fine-tuning.py 8 &

### TFA COCO ###
bash ./tools/detection/dist_train.sh configs/detection/tfa/coco/tfa_r101_fpn_coco_base-training.py 8
srun49 python3 -m tools.detection.misc.initialize_bbox_head --src1 ./work_dirs/tfa_r101_fpn_coco_base-training/latest.pth --method random_init --save-dir work_dirs/tfa_r101_fpn_coco_base-training --coco
srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning.py 8 &

srun49 bash ./tools/detection/dist_test.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py ./work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth 8 --eval mAP


### FSCE VOC ###
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_10shot-fine-tuning.py 8 &

srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_contrastive-loss_voc-split2_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split2/fsce_r101_fpn_voc-split2_10shot-fine-tuning.py 8 &

srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_1shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_2shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_3shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_5shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_contrastive-loss_voc-split3_10shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_3shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_5shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/voc/split3/fsce_r101_fpn_voc-split3_10shot-fine-tuning.py 8 &

### FSCE COCO ###
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.py 8  &
srun49 bash ./tools/detection/dist_train.sh configs/detection/fsce/coco/fsce_r101_fpn_coco_30shot-fine-tuning.py 8  &

### ATTENTION RPN COCO ###
bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py 4
bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training.py 4
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py 4 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_30shot-fine-tuning.py 4 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-10shot-fine-tuning.py 4 &

srun49 bash ./tools/detection/dist_test.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py ./work_dirs/attention-rpn_r50_c4_4xb2_coco_base-training/latest.pth 8 --eval mAP &
srun49 bash ./tools/detection/dist_test.sh configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_official-base-training.py ./work_dirs/attention-rpn_r50_c4_4xb2_coco_official-base-training/latest.pth 8 --eval mAP &

### ATTENTION RPN VOC ###
srun38 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_base-training.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_base-training.py 8 &

srun37 bash ./tools/detection/dist_test.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py upload/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training_20211101_003606-58a8f413.pth 8 --eval mAP &

srun38 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning.py 8 --options "load_from=upload/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training_20211101_003606-58a8f413.pth" &


srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_2shot-fine-tuning.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/attention-rpn_r50_c4_voc-split1_base-training/latest.pth" &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split2/attention-rpn_r50_c4_voc-split2_10shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_1shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_2shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_3shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_5shot-fine-tuning.py 8 &
srun48 bash ./tools/detection/dist_train.sh configs/detection/attention_rpn/voc/split3/attention-rpn_r50_c4_voc-split3_10shot-fine-tuning.py 8 &

### Meta-RCNN COCO ###
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_base-training.py 8

srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r50_c4_8xb4_coco_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_8xb4_coco_30shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r50_c4_8xb4_coco_base-training/latest.pth" &

srun49 bash ./tools/detection/dist_test.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py ./work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth 8 --eval mAP
### Meta-RCNN VOC ###
srun38 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_base-training.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_base-training.py 8 &

srun38g4 bash ./tools/detection/dist_test.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py  ./upload/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning_20211111_173217-b872c72a.pth 8 --eval mAP &


srun48 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8 --options "load_from=./upload/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training_20211101_234042-7184a596.pth" "seed=111" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &

srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split2/meta-rcnn_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split2_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split3_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split3_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split3_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split3_base-training/latest.pth"  &
srun37 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/voc/split3/meta-rcnn_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/meta-rcnn_r101_c4_8xb4_voc-split3_base-training/latest.pth"  &

### FSDetView COCO ###
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training.py 8

srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training.py 8 --work-dir ./work_dirs/fsdetview_r50_c4_coco_base-training_lr001
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_base-training.py 8 --work-dir ./work_dirs/fsdetview_r50_c4_coco_base-training_lr001_bbox32


python3 -m tools.detection.misc.initialize_checkpoint --src1 ./work_dirs/fsdetview_r50_c4_coco_base-training/latest.pth --method random_init --save-dir work_dirs/fsdetview_r50_c4_coco_base-training -coco


srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r50_c4_8xb4_coco_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_30shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r50_c4_8xb4_coco_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.py 8 &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/coco/fsdetview_r50_c4_8xb4_coco_30shot-fine-tuning.py 8 &


srun49 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_coco_10shot-fine-tuning.py 8 &
srun49 bash ./tools/detection/dist_train.sh configs/detection/meta_rcnn/coco/meta-rcnn_r50_c4_coco_30shot-fine-tuning.py 8 &

srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/fsdetview_r101_c4_8xb4_voc-split1_base-training_infin/latest.pth" &

### FSDetView VOC ###

srun48 bash ./tools/detection/dist_test.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training_20211101_072143-6d1fd09d.pth 8 --eval mAP &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py 8 --work-dir ./work_dirs/fsdetview_r101_c4_8xb4_voc-split1_base-training_infin &
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_base-training.py 8 --work-dir ./work_dirs/fsdetview_r101_c4_8xb4_voc-split2_base-training_infin &
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_base-training.py 8 --work-dir ./work_dirs/fsdetview_r101_c4_8xb4_voc-split3_base-training_infin &

srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun37 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &

srun48 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8 --options "load_from=https://download.openmmlab.com/mmfewshot/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training_20211101_072143-6d1fd09d.pth" &

srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split1_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split2/fsdetview_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split2_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split3_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split3_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split3_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split3_base-training/latest.pth" &
srun38 bash ./tools/detection/dist_train.sh configs/detection/fsdetview/voc/split3/fsdetview_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py 8 --options "load_from=./work_dirs/released_det/fsdetview_r101_c4_8xb4_voc-split3_base-training/latest.pth" &

### MPSR COCO ###
bash ./tools/detection/dist_train.sh configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training.py 2
srun38g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_base-training_20211103_164720-c6998b36.pth" &
export PORT=10086 && srun37g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/coco/mpsr_r101_fpn_2xb2_coco_30shot-fine-tuning.py 2 &

### MPSR VOC ###
srun37g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py 2 &  --work-dir ./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training_biaslrx2_4offset
srun37 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_8xb2_voc-split1_base-training.py 8 &
srun37 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_8xb2_voc-split2_base-training.py 8 &
srun37 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_8xb2_voc-split3_base-training.py 8 &

export PORT=10086 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py 2 &
export PORT=10087 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py 2 &
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py 2 &

srun49 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py 8 --options "load_from=./work_dirs/mpsr_r101_fpn_8xb2_voc-split1_base-training/latest.pth" &

srun49 bash ./tools/detection/dist_test.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py ./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_test.pth 8 --eval mAP &
srun48 bash ./tools/detection/dist_test.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py ./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2_test.pth 8 --eval mAP &
srun49 bash ./tools/detection/dist_test.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py ./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3_test.pth 8 --eval mAP &


export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth" &

export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_base-training.py 2 && \
export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth" && \
export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth" && \
export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth"  && \
export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth" && \
export PORT=10082 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split1/convert_mpsr_r101_fpn_2xb2_voc-split1_base-training-c186aaef.pth"

export PORT=10086 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_base-training.py 2 && \
export PORT=10086 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth" && \
export PORT=10087 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth" && \
export PORT=10087 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth" && \
export PORT=10087 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth" && \
export PORT=10087 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning.py 2 --options "load_from=upload/detection/mpsr/voc/split2/convert_mpsr_r101_fpn_2xb2_voc-split2_base-training-1861c370.pth" &

export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_base-training.py 2 && \
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning.py 2  --options "load_from=upload/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth" && \
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning.py 2  --options "load_from=upload/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth" && \
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning.py 2  --options "load_from=upload/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth" && \
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning.py 2  --options "load_from=upload/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth" && \
export PORT=10088 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning.py 2  --options "load_from=upload/detection/mpsr/voc/split3/convert_mpsr_r101_fpn_2xb2_voc-split3_base-training-1afa74d7.pth" &

export PORT=10011 && srun38g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_1shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_2.pth" && \
export PORT=10012 && srun38g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_2shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_2.pth" && \
export PORT=10013 && srun38g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_3shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_2.pth" && \
export PORT=10014 && srun38g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_5shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_2.pth" && \
export PORT=10015 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split1/mpsr_r101_fpn_2xb2_voc-split1_10shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split1_base-training/mpsr_voc_base_split1_2.pth" && \

export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_1shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_2shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_3shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_5shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split2/mpsr_r101_fpn_2xb2_voc-split2_10shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split2_base-training/mpsr_voc_base_split2.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_1shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_2shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_3shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_5shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3.pth" && \
export PORT=10011 && srun48g2 bash ./tools/detection/dist_train.sh configs/detection/mpsr/voc/split3/mpsr_r101_fpn_2xb2_voc-split3_10shot-fine-tuning.py 2 --options "load_from=./work_dirs/mpsr_r101_fpn_2xb2_voc-split3_base-training/mpsr_voc_base_split3.pth"
