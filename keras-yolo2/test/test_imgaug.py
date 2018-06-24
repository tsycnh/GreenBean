from imgaug import augmenters as iaa
import imgaug as ia
import cv2 as cv
import numpy as np
import random
import xml.etree.ElementTree as ET
from os import environ
environ['SCIPY_PIL_IMAGE_VIEWER'] = '/Applications/LilyView.app/Contents/MacOS/LilyView'
def load_anno(file):
    tree = ET.ElementTree(file=file)
    targets = []
    print(file)
    for elem in tree.iter(tag='object'):
        target = {}
        for child in elem:
            if child.tag == 'name':
                target['class'] = child.text
            if child.tag == 'bndbox':
                target['bndbox'] = {
                    'xmin':int(child.find('xmin').text),
                    'ymin':int(child.find('ymin').text),
                    'xmax':int(child.find('xmax').text),
                    'ymax':int(child.find('ymax').text),
                }
        targets.append(target)
    print(targets)
    return targets
# def paste(background,image,offset_x,offset_y):
#     background[offset_y:image.shape[0]+offset_y,# 竖直方向偏移
#                 offset_x:image.shape[1]+offset_x    # 水平方向偏移
#     ] = image
#     return background
# def func_images(images, random_state, parents, hooks):
#     print('进入func_images')
#     print('图片数量：',len(images))
#     global all_offsets
#     all_offsets= []
#     for i,image in enumerate(images):
#         img_rows,img_cols,img_channels = image.shape
#         bg_size = 0
#         if img_rows<img_cols:# 情况1
#             bg_size = img_cols
#             max_offset_y = bg_size - img_rows
#             offset_y = random.randint(0,max_offset_y)
#             offset_x = 0
#         elif img_rows>img_cols:# 情况2
#             bg_size = img_rows
#             max_offset_x = bg_size - img_cols
#             offset_x = random.randint(0,max_offset_x)
#             offset_y = 0
#         else:
#             offset_x = offset_y = 0
#         bg = np.zeros([bg_size,bg_size,img_channels],dtype=image.dtype)
#         bg = paste(bg,image,offset_x,offset_y)
#         images[i] = bg
#         all_offsets.append({
#             'offset_x':offset_x,
#             'offset_y':offset_y
#         })
#     return images
# def func_keypoints(keypoints_on_images, random_state, parents, hooks):
#     print('进入func_keypoints')
#     return keypoints_on_images
class SquarePad(iaa.Augmenter):
    """
    Augmenter that calls a lambda function for each batch of input image.

    This is useful to add missing functions to a list of augmenters.

    Parameters
    ----------
    func_images : callable,
        The function to call for each batch of images.
        It must follow the form

            ``function(images, random_state, parents, hooks)``

        and return the changed images (may be transformed in-place).
        This is essentially the interface of `Augmenter._augment_images()`.

    func_keypoints : callable,
        The function to call for each batch of image keypoints.
        It must follow the form

            ``function(keypoints_on_images, random_state, parents, hooks)``

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of `Augmenter._augment_keypoints()`.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    >>>     return keypoints_on_images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images,
    >>>     func_keypoints=func_keypoints
    >>> )

    Replaces every second row in images with black pixels and leaves keypoints
    unchanged.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(SquarePad, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.all_offsets = []

    def func_images(self,images, random_state, parents, hooks):
        print('进入func_images')
        print('图片数量：', len(images))

        for i, image in enumerate(images):
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
                offset_x = offset_y = 0
            bg = np.zeros([bg_size, bg_size, img_channels], dtype=image.dtype)
            bg = self.paste(bg, image, offset_x, offset_y)
            images[i] = bg
            self.all_offsets.append({
                'offset_x': offset_x,
                'offset_y': offset_y
            })
        return images

    def func_keypoints(self,keypoints_on_images, random_state, parents, hooks):
        print('进入func_keypoints')
        print(keypoints_on_images)
        for i in range(len(keypoints_on_images)):
            for j in range(len(keypoints_on_images[i].keypoints)):
                keypoints_on_images[i].keypoints[j].x += self.all_offsets[i]['offset_x']
                keypoints_on_images[i].keypoints[j].y += self.all_offsets[i]['offset_y']
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


if __name__ == '__main__':
    # 这个图像增强做以下几个事情：
    # 1. 将图像按照长边尺寸生成纯黑背景
    # 2. 将图像顺着短边方向往背景上随机粘贴。注意：标记数据也要跟着变

    # aug = iaa.Lambda(
    #     func_images=func_images,
    #     func_keypoints=func_keypoints
    # )

    seq = iaa.Sequential([
        SquarePad()
    ])

    imgs = []
    annos = []
    bbses = []
    clses = []
    for i in range(5):
        img1 = cv.imread('images/000%d.bmp'%(i+1))
        anno = load_anno('images/000%d.xml'%(i+1))
        imgs.append(img1)
        annos.append(anno)
        tmp = []
        tmp_cls = []
        for bbox in anno:
            tmp.append(ia.BoundingBox(
                x1=bbox['bndbox']['xmin'],
                y1=bbox['bndbox']['ymin'],
                x2=bbox['bndbox']['xmax'],
                y2=bbox['bndbox']['ymax']
            ))
            tmp_cls.append(bbox['class'])
        bbs = ia.BoundingBoxesOnImage(tmp,shape=img1.shape)
        bbses.append(bbs)
        clses.append(tmp_cls)
    seq_det = seq.to_deterministic()

    imgs_aug = seq_det.augment_images(imgs)
    bbses_aug = seq_det.augment_bounding_boxes(bbses)

    for i,img in enumerate(imgs_aug):
        image_before = bbses[i].draw_on_image(imgs[i])
        image_after = bbses_aug[i].draw_on_image(imgs_aug[i])
        cv.imshow('before'+str(i+1),image_before)
        cv.imshow('after'+str(i+1),image_after)

    cv.waitKey(0)