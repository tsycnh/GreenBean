from imgaug import augmenters as iaa
import imgaug as ia
import cv2 as cv
import numpy as np
import random
import cv2
import copy
from keras.utils import Sequence
from utils import BoundBox, bbox_iou
import keras
from os import environ
environ['SCIPY_PIL_IMAGE_VIEWER'] = '/Applications/LilyView.app/Contents/MacOS/LilyView'

class SquarePad(iaa.Augmenter):

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(SquarePad, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.all_offsets = []
        self.all_new_shapes=[]

    def func_images(self,images, random_state, parents, hooks):
        print('进入func_images')
        print('图片数量：', len(images))

        for i, image in enumerate(images):
            print('image index: ',i)
            img_rows, img_cols, img_channels = image.shape
            bg_size = 0
            if img_rows < img_cols:  # 情况1
                bg_size = img_cols
                max_offset_y = bg_size - img_rows
                offset_y = random.randint(0, max_offset_y)
                offset_x = 0
            elif img_rows > img_cols:  # 情况2
                bg_size = img_rows
                max_offset_x = bg_size - img_cols
                offset_x = random.randint(0, max_offset_x)
                offset_y = 0
            else:
                bg_size = img_rows
                offset_x = offset_y = 0
            bg = np.zeros([bg_size, bg_size, img_channels], dtype=image.dtype)
            bg = self.paste(bg, image, offset_x, offset_y)
            images[i] = bg
            self.all_offsets.append({
                'offset_x': offset_x,
                'offset_y': offset_y
            })
            self.all_new_shapes.append((bg_size,bg_size,img_channels))
        return images

    def func_keypoints(self,keypoints_on_images, random_state, parents, hooks):
        print('进入func_keypoints')
        print(keypoints_on_images)
        for i in range(len(keypoints_on_images)):
            for j in range(len(keypoints_on_images[i].keypoints)):
                keypoints_on_images[i].keypoints[j].x += self.all_offsets[i]['offset_x']
                keypoints_on_images[i].keypoints[j].y += self.all_offsets[i]['offset_y']
                keypoints_on_images[i].shape = self.all_new_shapes[i]
        return keypoints_on_images
    def _augment_images(self, images, random_state, parents, hooks):
        return self.func_images(images, random_state, parents=parents, hooks=hooks)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = self.func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks)
        assert isinstance(result, list)
        assert all([isinstance(el, ia.KeypointsOnImage) for el in result])
        return result

    def get_parameters(self):
        return []

    def paste(self,background, image, offset_x, offset_y):
        background[offset_y:image.shape[0] + offset_y,  # 竖直方向偏移
        offset_x:image.shape[1] + offset_x  # 水平方向偏移
        ] = image
        return background

class BatchGenerator_for_USTB(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.generator = None

        self.images_with_objs = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.self.config['IMAGE_H'], self.config['IMAGE_W']
        self.aug_pipe = iaa.Sequential([SquarePad(),iaa.Scale({"height": self.config['IMAGE_H'], "width": self.config['IMAGE_W']})])
        # if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images_with_objs)) / self.config['BATCH_SIZE']))
    def __getitem__(self, idx):
        #1. 确定当前batch在整个数据序列中的位置
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images_with_objs):
            r_bound = len(self.images_with_objs)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        #2.读取图像和label
        bbses = []
        imgs = []
        ys = []
        for i in range(l_bound,r_bound):
            img = cv.imread(self.images_with_objs[i]['filename'])
            tmp_bbox = []
            tmp_y = []
            for bbox in self.images_with_objs[i]['object']:
                tmp_bbox.append(ia.BoundingBox(
                    x1=bbox['xmin'],
                    y1=bbox['ymin'],
                    x2=bbox['xmax'],
                    y2=bbox['ymax']
                ))
                tmp_y.append(bbox['name'])
            bbs = ia.BoundingBoxesOnImage(tmp_bbox,shape=img.shape)
            imgs.append(img)
            bbses.append(bbs)
            ys.append(tmp_y)

        #3. 图像预处理
        qug_pipe_det = self.aug_pipe.to_deterministic()

        aug_imgs = qug_pipe_det.augment_images(imgs)
        aug_bbses = qug_pipe_det.augment_bounding_boxes(bbses)
        aug_clses = ys

        # for i, _ in enumerate(aug_imgs):
        #     image_before = bbses[i].draw_on_image(imgs[i])
        #     image_after = aug_bbses[i].draw_on_image(aug_imgs[i])
        #     cv.imshow('before' + str(i + 1), image_before)
        #     cv.imshow('after' + str(i + 1), image_after)
        # pass
        # cv.waitKey(0)
        #2. 循环每一张图，制造yolo专用标签
        for i in range(0,len(aug_imgs)):
            # augment input image and fix object's position and size
            # img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            img = aug_imgs[i]
            all_objs = []
            for j in range(0,len(aug_bbses[i].bounding_boxes)):
                all_objs.append({
                    'name':aug_clses[i][j],
                    'xmin':aug_bbses[i].bounding_boxes[j].x1,
                    'ymin':aug_bbses[i].bounding_boxes[j].y1,
                    'xmax':aug_bbses[i].bounding_boxes[j].x2,
                    'ymax':aug_bbses[i].bounding_boxes[j].y2,
                })

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                # for obj in all_objs:
                #     if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                #         cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                #         cv2.putText(img, obj['name'],
                #                     (obj['xmin']+2, obj['ymin']+12),
                #                     0, 1.2e-3 * img.shape[0],
                #                     (0,255,0), 2)

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        #print(' new batch created', idx)
        # 返回的数据是给model喂的。形式要和model定义的一样，第一个是输入，第二个是ground_truth输出。这个
        # 网络输入有两个，原始图像和bbox真值。之所以这样写，按照作者的说法是为了加快loss计算速度。
        return [x_batch, b_batch], y_batch

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images_with_objs)

    def load_annotation(self, i):
        annots = []

        for obj in self.images_with_objs[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images_with_objs[i]['filename'])

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images_with_objs)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        # for obj in all_objs:
        #     for attr in ['xmin', 'xmax']:
        #         if jitter: obj[attr] = int(obj[attr] * scale - offx)
        #
        #         obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
        #         obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
        #
        #     for attr in ['ymin', 'ymax']:
        #         if jitter: obj[attr] = int(obj[attr] * scale - offy)
        #
        #         obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
        #         obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)
        #
        #     if jitter and flip > 0.5:
        #         xmin = obj['xmin']
        #         obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
        #         obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs
