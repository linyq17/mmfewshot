#!/usr/bin/env bash
srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py &
srun37g1 bash ./tools/classification/dist_train.sh configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_cub_5way_1shot.py ./work_dirs/baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 bash ./tools/classification/dist_test.sh configs/classification/baseline/cub/baseline_conv4_cub_5way_1shot.py ./work_dirs/baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth

srun38g2 bash ./tools/classification/dist_train.sh configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py 2 &

srun37g2 bash ./tools/classification/dist_train.sh configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py 2 --work-dir "./work_dirs/test_baseline_2gpu" --cfg-options data.samples_per_gpu=32 &

srun37g2 bash ./tools/classification/dist_test.sh configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py https://download.openmmlab.com/mmfewshot/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot_20211120_095923-3a346523.pth 2  &

srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_cub_5way_5shot.py ./work_dirs/baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_cub_5way_1shot.py ./work_dirs/baseline_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_cub_5way_5shot.py ./work_dirs/baseline_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_cub_5way_1shot.py ./work_dirs/baseline-plus_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_cub_5way_5shot.py ./work_dirs/baseline-plus_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_cub_5way_1shot.py ./work_dirs/baseline-plus_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_cub_5way_5shot.py ./work_dirs/baseline-plus_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg_cosine_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg_cosine_conv4_cub_5way_1shot.py ./work_dirs/neg_cosine_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg_cosine_conv4_cub_5way_5shot.py ./work_dirs/neg_cosine_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg_cosine_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg_cosine_resnet12_cub_5way_1shot.py ./work_dirs/neg_cosine_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg_cosine_resnet12_cub_5way_5shot.py ./work_dirs/neg_cosine_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_cub_5way_1shot.py ./work_dirs/matching-net_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_cub_5way_5shot.py ./work_dirs/matching-net_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_cub_5way_1shot.py ./work_dirs/matching-net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_cub_5way_5shot.py ./work_dirs/matching-net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto_net_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_conv4_cub_5way_1shot.py ./work_dirs/proto_net_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_conv4_cub_5way_5shot.py ./work_dirs/proto_net_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto_net_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_resnet12_cub_5way_1shot.py ./work_dirs/proto_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_resnet12_cub_5way_5shot.py ./work_dirs/proto_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation_net_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_conv4_cub_5way_1shot.py ./work_dirs/relation_net_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_conv4_cub_5way_5shot.py ./work_dirs/relation_net_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation_net_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_resnet12_cub_5way_1shot.py ./work_dirs/relation_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_resnet12_cub_5way_5shot.py ./work_dirs/relation_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_cub_5way_1shot.py ./work_dirs/maml_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_cub_5way_5shot.py ./work_dirs/maml_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_cub_5way_1shot.py ./work_dirs/maml_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_cub_5way_5shot.py ./work_dirs/maml_resnet12_cub_5way_1shot/best_accuracy_mean.pth &


srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta_baseline_conv4_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta_baseline_conv4_cub_5way_1shot.py ./work_dirs/meta_baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta_baseline_conv4_cub_5way_5shot.py ./work_dirs/meta_baseline_conv4_cub_5way_1shot/best_accuracy_mean.pth &

srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta_baseline_resnet12_cub_5way_1shot.py &&
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta_baseline_resnet12_cub_5way_1shot.py ./work_dirs/meta_baseline_resnet12_cub_5way_1shot/best_accuracy_mean.pth &&
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta_baseline_resnet12_cub_5way_5shot.py ./work_dirs/meta_baseline_resnet12_cub_5way_1shot/best_accuracy_mean.pth &

#
#
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_wrn28x10_cub_5way_1shot.py
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-1shot.py ./work_dirs/baseline-plus_conv4_cub_5way_1shot/best_accuracy_mean.pth &
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-5shot.py ./work_dirs/baseline-plus_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-1shot.py ./work_dirs/baseline-plus_resnet12_cub_5way_1shot/best_accuracy_mean.pth &
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-5shot.py ./work_dirs/baseline-plus_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_wrn28x10_cub_5way_1shot.py ./work_dirs/baseline-plus_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_wrn28x10_cub_5way_5shot.py ./work_dirs/baseline-plus_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#
#
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_resnet18_cub_5way_1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_wrn28x10_cub_5way_1shot.py
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/matching-net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/matching-net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/matching-net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/matching-net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_wrn28x10_cub_5way_1shot.py ./work_dirs/matching-net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_wrn28x10_cub_5way_5shot.py ./work_dirs/matching-net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#
#
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto_net_resnet18_cub_5way_1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto_net_wrn28x10_cub_5way_1shot.py
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/proto_net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/proto_net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/proto_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/proto_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_wrn28x10_cub_5way_1shot.py ./work_dirs/proto_net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto_net_wrn28x10_cub_5way_5shot.py ./work_dirs/proto_net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation_net_resnet18_cub_5way_1shot.py
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation_net_wrn28x10_cub_5way_1shot.py
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/relation_net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/relation_net_conv4_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/relation_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/relation_net_resnet12_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_wrn28x10_cub_5way_1shot.py ./work_dirs/relation_net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation_net_wrn28x10_cub_5way_5shot.py ./work_dirs/relation_net_wrn28x10_cub_5way_1shot/best_accuracy_mean.pth
#
#
#srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py 8
#srun49g2 bash ./tools/classification/dist_train.sh configs/classification/baseline/mini_imagenet/baseline_wrn28x10_mini_imagenet_5way_1shot_2gpu.py 2
#srun48g2 bash ./tools/classification/dist_train.sh configs/classification/baseline_plus/mini_imagenet/baseline-plus_wrn28x10_mini_imagenet_5way_1shot_2gpu.py 2
#srun38g2 bash ./tools/classification/dist_train.sh configs/classification/neg_margin/mini_imagenet/neg_cosine_wrn28x10_mini_imagenet_5way_1shot_2gpu.py 2
#
#
#bash ./tools/classification/dist_train.sh configs/classification/baseline/cub/baseline_wrn28x10_cub_5way_1shot_4gpu.py 4
#srun49g4 bash ./tools/classification/dist_train.sh configs/classification/baseline/cub/baseline_wrn28x10_cub_5way_1shot_4gpu.py 4
#
#srun49g4 bash ./tools/classification/dist_train.sh configs/classification/
#
#srun49g4 bash ./tools/classification/dist_train.sh configs/classification/baseline/tiered_imagenet/baseline_conv4_tiered_imagenet_5way_1shot_4gpu.py 4
#
#srun49g4 bash ./tools/classification/dist_train.sh configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_tiered_imagenet_5way_1shot_4gpu.py 4
#bash ./tools/classification/dist_train.sh configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_tiered_imagenet_5way_1shot_4gpu.py 4
srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py ./work_dirs/baseline_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth  --show_task_results &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py ./work_dirs/baseline_conv4_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot.py ./work_dirs/baseline_resnet12_1xb64_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-5shot.py ./work_dirs/baseline_resnet12_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/baseline_conv4_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/baseline_conv4_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/baseline_resnet12_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/baseline_resnet12_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/baseline_conv4_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth --options "test.meta_test_cfg.support.num_workers=1" &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/baseline_conv4_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &


srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-1shot.py ./work_dirs/baseline-plus_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-5shot.py ./work_dirs/baseline-plus_conv4_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-1shot.py ./work_dirs/baseline-plus_resnet12_1xb64_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-5shot.py ./work_dirs/baseline-plus_resnet12_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/baseline-plus_conv4_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/baseline-plus_conv4_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/baseline-plus_resnet12_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/baseline-plus_resnet12_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/baseline-plus_conv4_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/baseline-plus_conv4_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &


srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-5shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-5shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-5shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./work_dirs/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &



srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py ./work_dirs/maml_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot.py ./work_dirs/maml_conv4_1xb105_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/maml_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/maml_resnet12_1xb105_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/maml_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/maml_conv4_1xb105_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/maml_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/maml_resnet12_1xb105_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/maml_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/maml_conv4_1xb105_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/maml_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/maml_resnet12_1xb105_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &




srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-5shot.py &

srun49g2 bash ./tools/classification/dist_test.sh configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/matching-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth 2 &

srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/matching-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/matching-net_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/matching-net_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/matching-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/matching-net_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/matching-net_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &



srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/proto-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/proto-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/proto-net_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/proto-net_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/proto-net_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/proto-net_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot.py &
#srun37g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py &

srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py ./work_dirs/relation-net_conv4_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot.py ./work_dirs/relation-net_conv4_1xb105_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py ./work_dirs/relation-net_resnet12_1xb105_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot.py ./work_dirs/relation-net_resnet12_1xb105_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/relation-net_conv4_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/relation-net_conv4_1xb105_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./work_dirs/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./work_dirs/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./work_dirs/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./work_dirs/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &



srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot.py &
srun38g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot.py &
srun37g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot.py &


srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py ./work_dirs/meta-baseline_conv4_1xb100_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot.py ./work_dirs/meta-baseline_conv4_1xb100_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot.py ./work_dirs/meta-baseline_resnet12_1xb100_cub_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot.py ./work_dirs/meta-baseline_resnet12_1xb100_cub_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot.py ./work_dirs/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot.py ./work_dirs/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot.py ./work_dirs/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot.py ./work_dirs/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot.py ./work_dirs/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot.py ./work_dirs/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot.py ./work_dirs/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot/best_accuracy_mean.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot.py ./work_dirs/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot/best_accuracy_mean.pth &


srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot.py ./upload/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-1shot_20211031_122455-f172187a.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-5shot.py ./upload/classification/neg_margin/cub/neg-margin_cosine_conv4_1xb64_cub_5way-5shot_20211031_122632-1b236969.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot.py ./upload/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-1shot_20211031_122825-5243c4c4.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-5shot.py ./upload/classification/neg_margin/cub/neg-margin_cosine_resnet12_1xb64_cub_5way-5shot_20211031_123026-ac9c99f3.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-5shot_20211102_204224-4dee0f42.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/neg_margin/tiered_imagenet/neg-margin_cosine_conv4_1xb64_tiered-imagenet_5way-1shot_20211102_201102-ad0dcc4e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-5shot_20211102_215227-4a28a26e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/neg_margin/tiered_imagenet/neg-margin_cosine_resnet12_1xb64_tiered-imagenet_5way-1shot_20211102_212208-00ace9d6.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-1shot_20211102_124805-45689059.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-5shot_20211102_144530-d3b3f147.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/neg_margin/mini_imagenet/neg-margin_cosine_resnet12_1xb64_mini-imagenet_5way-1shot_20211102_131328-03e8f6b1.pth &
srun37g1 python ./tools/classification/test.py configs/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/neg_margin/mini_imagenet/neg-margin_cosine_conv4_1xb64_mini-imagenet_5way-5shot_20211102_130234-4fbf53a6.pth &

srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-5shot.py ./upload/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-5shot_20211031_113731-b95c259c.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-1shot.py ./upload/classification/baseline_plus/cub/baseline-plus_resnet12_1xb64_cub_5way-1shot_20211031_113731-0fe5b21c.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-5shot.py ./upload/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-5shot_20211031_113731-9d828ca8.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-1shot.py ./upload/classification/baseline_plus/cub/baseline-plus_conv4_1xb64_cub_5way-1shot_20211031_113731-9d4e861d.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-5shot_20211101_123338-b09e587c.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-5shot_20211102_041355-92d6f804.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/baseline_plus/tiered_imagenet/baseline-plus_resnet12_1xb64_tiered-imagenet_5way-1shot_20211101_192517-277f50b2.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/baseline_plus/tiered_imagenet/baseline-plus_conv4_1xb64_tiered-imagenet_5way-1shot_20211101_113621-52e05047.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-1shot_20211101_033022-627a6a7e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/baseline_plus/mini_imagenet/baseline-plus_conv4_1xb64_mini-imagenet_5way-5shot_20211101_034545-d62e7692.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-1shot_20211101_051442-6956610a.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/baseline_plus/mini_imagenet/baseline-plus_resnet12_1xb64_mini-imagenet_5way-5shot_20211101_103756-f426dc4f.pth &

srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot.py ./upload/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-5shot_20211031_174004-333d4764.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot.py ./upload/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-5shot_20211031_172930-292d5226.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot.py ./upload/classification/proto_net/cub/proto-net_resnet12_1xb105_cub_5way-1shot_20211031_173524-f5b74dcc.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot.py ./upload/classification/proto_net/cub/proto-net_conv4_1xb105_cub_5way-1shot_20211031_172601-c4670089.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-5shot_20211105_042550-87176760.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/proto_net/tiered_imagenet/proto-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211105_032924-54a83f60.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-5shot_20211104_233708-e859fe85.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/proto_net/tiered_imagenet/proto-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211104_210547-4090e593.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-5shot_20211104_210124-d1f3dd18.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/proto_net/mini_imagenet/proto-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211104_205713-ed4adb11.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-1shot_20211104_201720-55b7aa99.pth &
srun37g1 python ./tools/classification/test.py configs/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/proto_net/mini_imagenet/proto-net_conv4_1xb105_mini-imagenet_5way-5shot_20211104_202429-8ed4d592.pth &

srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot.py ./upload/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-1shot_20211031_113731-ace1fb85.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot.py ./upload/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-1shot_20211031_113731-3523ba7a.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-5shot.py ./upload/classification/baseline/cub/baseline_resnet12_1xb64_cub_5way-5shot_20211031_113731-712d99b1.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot.py ./upload/classification/baseline/cub/baseline_conv4_1xb64_cub_5way-5shot_20211031_113731-2b72fc93.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-5shot_20211101_031229-9ab9b6c6.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-5shot.py ./upload/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-5shot_20211101_030438-e714e654.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/baseline/tiered_imagenet/baseline_conv4_1xb64_tiered-imagenet_5way-1shot_20211031_230209-73195538.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot.py ./upload/classification/baseline/tiered_imagenet/baseline_resnet12_1xb64_tiered-imagenet_5way-1shot_20211101_031229-9515636a.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-5shot_20211031_214913-0f06c3b0.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/baseline/mini_imagenet/baseline_conv4_1xb64_mini-imagenet_5way-1shot_20211031_214913-9961b012.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot.py ./upload/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-1shot_20211031_220736-494e8f85.pth &
srun37g1 python ./tools/classification/test.py configs/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-5shot.py ./upload/classification/baseline/mini_imagenet/baseline_resnet12_1xb64_mini-imagenet_5way-5shot_20211031_225358-bf346fa2.pth &

srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot.py ./upload/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-5shot_20211031_213956-2d6e9d28.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot.py ./upload/classification/relation_net/cub/relation-net_resnet12_1xb105_cub_5way-1shot_20211031_213930-9265a565.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot.py ./upload/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-5shot_20211031_184353-a917723e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot.py ./upload/classification/relation_net/cub/relation-net_conv4_1xb105_cub_5way-1shot_20211031_183527-61bbf6ed.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-5shot_20211104_193011-2471cf0e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211104_124044-f1864279.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/relation_net/tiered_imagenet/relation-net_conv4_1xb105_tiered-imagenet_5way-5shot_20211104_132219-c8f5e78e.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/relation_net/tiered_imagenet/relation-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211104_152741-7d0dbf47.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-5shot_20211104_115939-027db65d.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-1shot_20211104_101509-2f719555.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/relation_net/mini_imagenet/relation-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211104_115657-28ce2cc7.pth &
srun37g1 python ./tools/classification/test.py configs/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/relation_net/mini_imagenet/relation-net_conv4_1xb105_mini-imagenet_5way-5shot_20211104_111809-0278fc06.pth &

srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot.py ./upload/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-1shot_20211104_003717-79040109.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot.py ./upload/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-1shot_20211104_003717-9b09ec20.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot.py ./upload/classification/meta_baseline/cub/meta-baseline_conv4_1xb100_cub_5way-5shot_20211104_003717-fd4aa5cc.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot.py ./upload/classification/meta_baseline/cub/meta-baseline_resnet12_1xb100_cub_5way-5shot_20211104_003717-bd92afc1.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot.py ./upload/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-5shot_20211104_061204-9d6a3e1d.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot.py ./upload/classification/meta_baseline/tiered_imagenet/meta-baseline_resnet12_1xb100_tiered-imagenet_5way-1shot_20211104_061204-234fcc40.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot.py ./upload/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-5shot_20211104_051558-8b11b0be.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot.py ./upload/classification/meta_baseline/tiered_imagenet/meta-baseline_conv4_1xb100_tiered-imagenet_5way-1shot_20211104_051313-6d2a56db.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot.py ./upload/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-1shot_20211104_045558-f7236038.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot.py ./upload/classification/meta_baseline/mini_imagenet/meta-baseline_resnet12_1xb100_mini-imagenet_5way-5shot_20211104_045731-5951dfab.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot.py ./upload/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-5shot_20211104_003717-64518831.pth &
srun37g1 python ./tools/classification/test.py configs/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot.py ./upload/classification/meta_baseline/mini_imagenet/meta-baseline_conv4_1xb100_mini-imagenet_5way-1shot_20211104_003717-ff36abc6.pth &

srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot.py ./upload/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-5shot_20211031_132342-dbc7ced0.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot.py ./upload/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-5shot_20211031_141356-2fd83668.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot.py ./upload/classification/matching_net/cub/matching-net_conv4_1xb105_cub_5way-1shot_20211031_131549-e69fa1d8.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot.py ./upload/classification/matching_net/cub/matching-net_resnet12_1xb105_cub_5way-1shot_20211031_140707-bddc6efb.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-1shot_20211103_173253-a00f7d49.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/matching_net/tiered_imagenet/matching-net_conv4_1xb105_tiered-imagenet_5way-5shot_20211103_211008-4ff37a28.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-5shot_20211104_063933-a18e59ef.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/matching_net/tiered_imagenet/matching-net_resnet12_1xb105_tiered-imagenet_5way-1shot_20211104_024200-0fb22915.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-5shot_20211102_232627-206e2763.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-5shot_20211103_102428-7a07abfe.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/matching_net/mini_imagenet/matching-net_conv4_1xb105_mini-imagenet_5way-1shot_20211107_164532-dbd5300d.pth &
srun37g1 python ./tools/classification/test.py configs/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/matching_net/mini_imagenet/matching-net_resnet12_1xb105_mini-imagenet_5way-1shot_20211103_065857-6dbdb299.pth &

srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot.py ./upload/classification/maml/cub/maml_conv4_1xb105_cub_5way-5shot_20211031_125919-48d15b7a.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot.py ./upload/classification/maml/cub/maml_resnet12_1xb105_cub_5way-5shot_20211031_130534-1de7c6d3.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot.py ./upload/classification/maml/cub/maml_resnet12_1xb105_cub_5way-1shot_20211031_130240-542c4387.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot.py ./upload/classification/maml/cub/maml_conv4_1xb105_cub_5way-1shot_20211031_125617-c00d532b.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-5shot_20211104_110734-88a559bc.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/maml/tiered_imagenet/maml_conv4_1xb105_tiered-imagenet_5way-1shot_20211104_110055-a08f6d09.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot.py ./upload/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-5shot_20211104_115442-95920719.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot.py ./upload/classification/maml/tiered_imagenet/maml_resnet12_1xb105_tiered-imagenet_5way-1shot_20211104_115442-340cb23f.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-5shot_20211104_105516-13547c7b.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-1shot_20211104_072004-02e6c7a7.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot.py ./upload/classification/maml/mini_imagenet/maml_conv4_1xb105_mini-imagenet_5way-5shot_20211104_072004-4c336eec.pth &
srun37g1 python ./tools/classification/test.py configs/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot.py ./upload/classification/maml/mini_imagenet/maml_resnet12_1xb105_mini-imagenet_5way-1shot_20211104_105317-d1628e14.pth &
