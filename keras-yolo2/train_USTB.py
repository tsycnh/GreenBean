#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf

    with open(config_path,encoding='UTF-8') as config_buffer:
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
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels_count.keys()
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    yolo.train(train_imgs         = train_imgs_with_objs,
               valid_imgs         = valid_imgs_with_objs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'])

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
