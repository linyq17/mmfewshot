# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import random

import mmcv
import numpy as np
from terminaltables import AsciiTable

import mmfewshot  # noqa: F401, F403
from mmfewshot.detection.datasets import (COCO_SPLIT, VOC_SPLIT, NumpyEncoder,
                                          build_dataset)
from .visualize_saved_dataset import Visualizer

VOC = dict(
    type='FewShotVOCDataset',
    ann_file=[
        'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        'data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    ],
    img_prefix='data/VOCdevkit/',
    pipeline=[],
    classes=[],
    use_difficult=False,
    instance_wise=False)

COCO = dict(
    type='FewShotCocoDataset',
    ann_file='data/few_shot_ann/coco/annotations/train.json',
    img_prefix='data/coco/',
    pipeline=[],
    classes=[],
    instance_wise=False,
)


class DatasetGenerator(object):

    def __init__(self,
                 dataset,
                 output_path,
                 setting,
                 novel_classes=None,
                 base_classes=None,
                 novel_shot=1,
                 base_shot=1,
                 seed=None):
        assert base_classes is not None or \
               novel_classes is not None, 'require classes labels'
        self.output_path = output_path
        self.dataset = dataset
        self.data_infos = dataset.data_infos
        self.novel_classes = novel_classes \
            if novel_classes is not None else []
        self.base_classes = base_classes \
            if base_classes is not None else []
        self.novel_shot = novel_shot
        self.base_shot = base_shot
        self.times = 0
        self.setting = setting
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.shot_per_image_stat = None
        self.bbox_area_stat = None
        self.area_bin = [16, 32, 64, 96, 128, 160, 192, 224, 256, 1000]
        self.total_instance = 0
        self.get_stat()
        self.print_stat()

    def generate_few_shot_dataset(self):
        self.times += 1
        sampled_data_infos = self.sample_shot()
        save_path = os.path.join(self.output_path,
                                 f'{self.setting}_t{self.times}.json')
        meta_info = [{
            'CLASSES': self.dataset.CLASSES,
            'img_prefix': self.dataset.img_prefix
        }]
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                meta_info + sampled_data_infos, f, indent=4, cls=NumpyEncoder)
        return save_path

    def sample_shot(self):
        novel_classes_pool = {
            class_name: []
            for class_name in self.novel_classes
        }
        base_only_classes_pool = []
        for idx, data_info in enumerate(self.data_infos):
            ann = data_info['ann']
            num_bbox = ann['labels'].shape[0]
            skip = False
            has_novel = False
            num_shots_per_classes = \
                {class_name: 0 for class_name in self.novel_classes}
            for i in range(num_bbox):
                c = self.dataset.CLASSES[ann['labels'][i]]
                if c in self.novel_classes:
                    num_shots_per_classes[c] += 1
                    has_novel = True
                    if num_shots_per_classes[c] > 1:
                        skip = True
            if skip:
                continue
            if not has_novel:
                base_only_classes_pool.append(idx)
            else:
                for c in self.novel_classes:
                    if num_shots_per_classes[c] > 0:
                        novel_classes_pool[c].append(idx)
        total_sampled_idx = []
        count_novel_shot = {class_name: 0 for class_name in self.novel_classes}
        # add novel classes shot
        for i in range(self.novel_shot):
            for c in self.novel_classes:
                if count_novel_shot[c] > i:
                    continue
                while True:
                    seleced_idx = np.random.choice(novel_classes_pool[c])
                    if seleced_idx not in total_sampled_idx:
                        labels = self.data_infos[seleced_idx]['ann']['labels']
                        unique, counts = np.unique(labels, return_counts=True)
                        skip = False
                        for j in range(unique.shape[0]):
                            c = self.dataset.CLASSES[unique[j]]
                            if c in self.novel_classes and \
                                    count_novel_shot[c] \
                                    + counts[j] > self.novel_shot:
                                skip = True
                                break
                        if not skip:
                            for j in range(unique.shape[0]):
                                c = self.dataset.CLASSES[unique[j]]
                                if c in self.novel_classes:
                                    count_novel_shot[c] += counts[j]
                            total_sampled_idx.append(seleced_idx)
                            break
        return [self.data_infos[i] for i in total_sampled_idx]

    def get_stat(self):
        # count 1 ~ 10 shot images, >10 shot images, total shots
        self.shot_per_image_stat = {
            class_name: [0 for _ in range(12)]
            for class_name in self.base_classes + self.novel_classes
        }
        # count bbox fall in area bin size for each classes
        self.bbox_area_stat = {
            class_name: [0 for _ in self.area_bin]
            for class_name in self.base_classes + self.novel_classes
        }
        for data_info in self.data_infos:
            num_shots_per_classes = {
                class_name: 0
                for class_name in self.base_classes + self.novel_classes
            }
            ann = data_info['ann']
            num_bbox = ann['labels'].shape[0]
            self.total_instance += num_bbox
            for i in range(num_bbox):
                c = self.dataset.CLASSES[ann['labels'][i]]
                bbox = ann['bboxes'][i]
                num_shots_per_classes[c] += 1
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                for j, size in enumerate(self.area_bin):
                    if area < size * size:
                        self.bbox_area_stat[c][j] += 1
                        break
            for c in self.base_classes + self.novel_classes:
                num_shots = num_shots_per_classes[c]
                if num_shots > 0:
                    if num_shots <= 10:
                        self.shot_per_image_stat[c][num_shots - 1] += 1
                    else:
                        self.shot_per_image_stat[c][10] += 1
                    self.shot_per_image_stat[c][11] += num_shots

    def print_stat(self):
        """Print the number of instance number."""
        result = (f'\ndataset statistics with number of images '
                  f'{len(self.data_infos)} and '
                  f'instance counts {self.total_instance} \n')
        if self.novel_classes is None and self.base_classes is None:
            result += 'Category names are not provided. \n'
            return result
        table_head = ['base category', 'novel category']
        for i, classes in enumerate([self.base_classes, self.novel_classes]):
            if len(classes) == 0:
                continue
            # create a table counting image shot
            result += '\n[' + table_head[i] + \
                      '] number of images with different shots\n'
            table_data = [['category'] + [f'{i + 1}' for i in range(10)] +
                          ['>10', 'total shots', 'total images']]
            for cls in classes:
                row_data = [f'{cls}']
                row_data += [
                    f'{self.shot_per_image_stat[cls][k]}' for k in range(11)
                ]
                row_data += [sum(self.shot_per_image_stat[cls][:10])]
                row_data += [self.shot_per_image_stat[cls][-1]]
                table_data.append(row_data)
            table = AsciiTable(table_data)
            result += table.table
            result += '\n[' + table_head[i] + \
                      '] number of bbox with different area < (size * size)\n'
            # create a table counting bbox area
            table_data = [['category'] +
                          [f'{size}' for size in self.area_bin[:-1]] +
                          [f'>{self.area_bin[-2]}']]
            for cls in classes:
                row_data = [f'{cls}']
                row_data += [
                    f'{self.bbox_area_stat[cls][i]}'
                    for i in range(len(self.area_bin))
                ]
                table_data.append(row_data)
            table = AsciiTable(table_data)
            result += table.table
        print(result)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a FewShot Dataset')
    parser.add_argument(
        '--setting',
        choices=['voc1', 'voc2', 'voc3', 'coco'],
        help='output dir to save visualize images')
    parser.add_argument(
        '--dir',
        default='./work_dirs/datasets',
        type=str,
        help='dir to save datasets')
    parser.add_argument(
        '--name', default='', type=str, help='name of dataset prefix')
    parser.add_argument(
        '--times', default=10, type=int, help='number of datasets to generate')
    parser.add_argument(
        '--novel-shot', default=1, type=int, help='number of novel shot')
    parser.add_argument(
        '--base-shot', default=1, type=int, help='number of novel shot')
    parser.add_argument(
        '--vis', action='store_true', help='whether to vis saved datasets.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.setting in ['voc1', 'voc2', 'voc3']:
        split = args.setting[-1]
        dataset_cfg = VOC
        dataset_cfg['classes'] = VOC_SPLIT[f'ALL_CLASSES_SPLIT{split}']
        novel_classes = VOC_SPLIT[f'NOVEL_CLASSES_SPLIT{split}']
        base_classes = VOC_SPLIT[f'BASE_CLASSES_SPLIT{split}']
    elif args.setting == 'coco':
        dataset_cfg = COCO
        dataset_cfg['classes'] = COCO_SPLIT['ALL_CLASSES']
        novel_classes = COCO_SPLIT['NOVEL_CLASSES']
        base_classes = COCO_SPLIT['BASE_CLASSES']
    else:
        raise ValueError('not defined settings')
    dataset = build_dataset(dataset_cfg, task_type='mmdet')
    work_dir = os.path.join(args.dir, args.name)
    mmcv.mkdir_or_exist(os.path.abspath(work_dir))
    g = DatasetGenerator(
        dataset,
        work_dir,
        setting=args.setting,
        novel_classes=novel_classes,
        base_classes=base_classes,
        novel_shot=args.novel_shot,
        base_shot=args.base_shot)
    for t in range(args.times):
        save_file = g.generate_few_shot_dataset()
        print(f'{t}: dataset is saved to {save_file}.')
        if args.vis:
            visualizer = Visualizer(save_file, work_dir)
            visualizer.visualize()
