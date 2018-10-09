#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json
from st_utils import BatchGenerator_for_USTB,draw_detections

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config_path = './config_USTB.json'
    weight_path = './full_yolo_USTB_7种缺陷_实验1.h5'

    with open(config_path, encoding='UTF-8') as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    # parse annotations of the training set
    train_imgs_with_objs, train_labels_count = parse_annotation(config['train']['train_annot_folder'],
                                                                config['train']['train_image_folder'],
                                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set

    valid_imgs_with_objs, valid_labels_count = parse_annotation(config['valid']['valid_annot_folder'],
                                                                config['valid']['valid_image_folder'],
                                                                config['model']['labels'])

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels_count.keys()))

        print('Seen labels:\t', train_labels_count)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels_count.keys()

    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    yolo.load_weights(weight_path)
    generator_config = {
                'IMAGE_H'         : config['model']['input_size'],
                'IMAGE_W'         : config['model']['input_size'],
                'GRID_H'          : 13,
                'GRID_W'          : 13,
                'BOX'             : len(config['model']['anchors'])//2,#取整运算
                'LABELS'          : config['model']['labels'],
                'CLASS'           : len(config['model']['labels']),
                'ANCHORS'         : config['model']['anchors'],
                'BATCH_SIZE'      : config['train']['batch_size'],
                'TRUE_BOX_BUFFER' : config['model']['max_box_per_image'],
            }
    valid_generator = BatchGenerator_for_USTB(valid_imgs_with_objs,generator_config,
        norm = yolo.feature_extractor.normalize,
        jitter = False)

    average_precisions = yolo.evaluate(valid_generator,save_path='detection_result_20181009')

    # print evaluation
    for label, average_precision in average_precisions.items():
        print(yolo.labels[label], '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

