from preprocessing import parse_annotation,BatchGenerator
from st_utils import BatchGenerator_for_USTB
import json
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

if __name__ == "__main__":
    config_path = './config_USTB.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])
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
    generator = BatchGenerator_for_USTB(images=train_imgs,config=generator_config,shuffle=False)

    def test_load_image():
        result = generator.load_image(2)
        # cv2.imshow('img_0', result['aug']['image'])
        print(result['aug']['annotation'])
    def test_getitem():
    # print(train_imgs,train_labels)
        d = generator.__getitem__(0,debug=False)
        image = d[0][0][2].astype('uint8')
        cv2.imshow('output',image)


        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=25, x2=75, y1=25, y2=75),
            ia.BoundingBox(x1=100, x2=150, y1=25, y2=75)
        ], shape=image.shape)
        bbs.on(image)
        #绘制bbox需要先project on 已padding的黑边图像，然后再on到原图
    # test load image

    # test_getitem()
    test_load_image()

    cv2.waitKey(0)

