"""This is the same code as in the demo notebook"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd

net = build_ssd('test', size=300, num_classes=2)  # initialize SSD
net.load_weights('../weights/orca/ORCA.pth')

from matplotlib import pyplot as plt
# from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import OrcaDetection, ORCA_ROOT, OrcaAnnotationTransform

# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = OrcaDetection(ORCA_ROOT, [('2007', 'val')], None, OrcaAnnotationTransform())
img_id = 177

if img_id < len(testset.ids):
    image = testset.pull_image(img_id)
    anno_boxes = testset.pull_anno(img_id)[1]
else:
    image = testset.pull_test_image(img_id)


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
# plt.figure(figsize=(10, 10))
# plt.imshow(rgb_image)
# currentAxis = plt.gca()
# pt = anno_boxes[0]
# coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
# coords_for_test = (50, 50), 100, 100
# # the (0, 0) point actually start from the top left corner

# currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='blue', linewidth=2))
# currentAxis.add_patch(plt.Rectangle(*coords_for_test, fill=False, edgecolor='white', linewidth=2))
# plt.show()

x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
# if torch.cuda.is_available():
#     xx = xx.cuda()
y = net(xx)

from data import ORCA_CLASSES as labels

top_k = 10

plt.figure(figsize=(10, 10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0, i, j, 0] >= 0.6:
        score = detections[0, i, j, 0]
        label_name = labels[i - 1]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        # so here is transforming it to the way that coords understands?
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        j += 1

if anno_boxes is not None:
    for pt_gold in anno_boxes:
        # pt_gold = anno_boxes[x]
        coords_gold = (pt_gold[0], pt_gold[1]), pt_gold[2] - pt_gold[0] + 1, pt_gold[3] - pt_gold[1] + 1
        currentAxis.add_patch(plt.Rectangle(*coords_gold, fill=False, edgecolor='white', linewidth=2))
        currentAxis.text(pt_gold[0], pt_gold[1], 'ground truth', bbox={'facecolor': 'white', 'alpha': 0.5})

plt.show()
