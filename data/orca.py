"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

ORCA_CLASSES = ('orca',)
ORCA_ROOT = '/media/liah/DATA/others/data/orcatag'

IMGS, ANNOS, CATES = 'images', 'annotations', 'categories'


class OrcaAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(ORCA_CLASSES, range(len(ORCA_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, img_id, img2annos, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for anno in img2annos[img_id]:
            xmin = anno['bbox'][1]
            ymin = anno['bbox'][0]
            xmax = anno['bbox'][1] + anno['bbox'][2] - 1
            ymax = anno['bbox'][0] + anno['bbox'][3] - 1
            bndbox = [xmin/width, ymin/height, xmax/width, ymax/height, anno['cate_id']]
            res.append(bndbox)

        # for obj in target.iter('object'):
        #     difficult = int(obj.find('difficult').text) == 1
        #     if not self.keep_difficult and difficult:
        #         continue
        #     name = obj.find('name').text.lower().strip()
        #     bbox = obj.find('bndbox')
        #
        #     pts = ['xmin', 'ymin', 'xmax', 'ymax']
        #     bndbox = []
        #     for i, pt in enumerate(pts):
        #         cur_pt = int(bbox.find(pt).text) - 1
        #         # scale height or width
        #         cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
        #         bndbox.append(cur_pt)
        #     label_idx = self.class_to_ind[name]
        #     bndbox.append(label_idx)
        #     res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        #     # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class OrcaDetection(data.Dataset):
    """Orca Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=OrcaAnnotationTransform(),
                 dataset_name='orcatag'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = osp.join(root, 'orca_object_anno.json')
        # images: list of dicts: {'file_name': {str}, 'id':{int}, 'height':{int}, 'width':{int}}
        # annotations: list of dicts: {'bbox': {List[int]}, 'category_id': {int}}, 'id': {int}, 'image_id':{int}}
        images, annotations, categories = self.load_annotation_json(json_file=self._annopath)
        self.ids = []
        self.imgid2imgfpath = {}
        self.imgid2imghw = {}
        self.imgid2annos = defaultdict(list)
        for anno in annotations:
            self.imgid2annos[anno['image_id']].append({'cate_id': anno['category_id'], 'bbox': anno['bbox']})
        for img in images:
            fpath = Path(self.root).joinpath('JPEGImages', img['file_name'])
            if fpath.is_file():
                self.ids.append(img['id'])
                self.imgid2imgfpath[img['id']] = Path(self.root).joinpath('JPEGImages', img['file_name'])
                self.imgid2imghw[img['id']] = (img['height'], img['width'])

        # for (year, name) in image_sets:
        #     # rootpath = osp.join(self.root, 'VOC' + year)
        #     rootpath = self.root
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         # todo self.ids are the image ids that contains this object class?
        #         self.ids.append((rootpath, line.strip().split()[0]))

    def load_annotation_json(self, json_file):
        with Path(json_file).open('r') as f:
            data = json.load(f)
            images, annotations, categories = data[IMGS], data[ANNOS], data[CATES]
        return images, annotations, categories

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self.imgid2imgfpath[img_id].as_posix())
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(img_id, self.imgid2annos, width, height)
            # now target is the scaled [xmin ymin xmax yman class_id]

        # self.transform = SSDAugmentation
        if self.transform is not None:
            #
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self.imgid2imgfpath[img_id].as_posix(), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(img_id, self.imgid2annos, 1, 1)
        # return img_id[1], gt
        return str(img_id), gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == "__main__":
    orca_dataset = OrcaDetection(root=ORCA_ROOT)
