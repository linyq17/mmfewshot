srun38g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/baseline_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/baseline_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/baseline_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline/tiered_imagenet/baseline_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/baseline_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/baseline_pp_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/baseline_pp_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/baseline_pp_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/baseline_pp/tiered_imagenet/baseline_pp_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/baseline_pp_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/neg_cosine_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/neg_cosine_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/neg_cosine_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/neg_margin/tiered_imagenet/neg_cosine_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/neg_cosine_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching_net_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching_net_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/matching_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching_net_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/matching_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/matching_net/tiered_imagenet/matching_net_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching_net_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/matching_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/matching_net/tiered_imagenet/matching_net_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/matching_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto_net_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto_net_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/proto_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto_net_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/proto_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/proto_net/tiered_imagenet/proto_net_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto_net_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/proto_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/proto_net/tiered_imagenet/proto_net_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/proto_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation_net_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation_net_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/relation_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation_net_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/relation_net_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/relation_net/tiered_imagenet/relation_net_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation_net_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/relation_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/relation_net/tiered_imagenet/relation_net_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/relation_net_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun49g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_conv4_tiered_imagenet_5way_1shot.py &&
srun49g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/maml_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &
srun49g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/maml_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun49g1 python ./tools/classification/train.py configs/classification/maml/tiered_imagenet/maml_resnet12_tiered_imagenet_5way_1shot.py &&
srun49g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/maml_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun49g1 python ./tools/classification/test.py configs/classification/maml/tiered_imagenet/maml_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/maml_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &


srun49g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_conv4_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_conv4_tiered_imagenet_5way_1shot.py ./work_dirs/meta_baseline_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_conv4_tiered_imagenet_5way_5shot.py ./work_dirs/meta_baseline_conv4_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &

srun38g1 python ./tools/classification/train.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_resnet12_tiered_imagenet_5way_1shot.py &&
srun38g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_resnet12_tiered_imagenet_5way_1shot.py ./work_dirs/meta_baseline_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &&
srun38g1 python ./tools/classification/test.py configs/classification/meta_baseline/tiered_imagenet/meta_baseline_resnet12_tiered_imagenet_5way_5shot.py ./work_dirs/meta_baseline_resnet12_tiered_imagenet_5way_1shot/best_accuracy_mean.pth &
