from http.server import BaseHTTPRequestHandler, HTTPServer

import base64
import json
# from PIL import Image
# import skimage.transform
# import io
import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import cv2

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathlib import Path
from ssd import build_ssd
net = build_ssd('test', size=300, num_classes=2)  # initialize SSD
net.load_weights('/weights/ORCA.pth')
from matplotlib import pyplot as plt
from data import OrcaDetection, ORCA_ROOT, OrcaAnnotationTransform
from data import ORCA_CLASSES as labels


__PORT__ = int(os.getenv('PORT', 8000))
__ADDRESS__ = os.getenv('ADDRESS', 'localhost')


# class ModelServer():
#     def __init__(self):
#         x = np.linspace(0, 1, 224)
#         y = np.linspace(0, 1, 224)
#         self.xx, self.yy = np.meshgrid(x, y)
#         self.xx, self.yy = self.xx[np.newaxis], self.yy[np.newaxis]

#     def predict(self, image):
#         height, width, channels= image.shape
#         print('Original shape={}'.format(image.shape))

#         image = skimage.transform.resize(image, (224, 224, 3), mode='constant', anti_aliasing=True)
#         image = np.moveaxis(image, 2, 0)
#         image = np.concatenate([self.xx, self.yy, image])
#         image = image[np.newaxis]
#         image = image

#         sample = torch.Tensor(image)
#         result = net(sample)[0]
        
#         ''' Move result coordinates back into original image's coordinate frame '''
#         print('result in resized coordinates={}'.format(result))
#         result[:, 0] = (result[:, 0] / 224.0) * width
#         result[:, 1] = (result[:, 1] / 224.0) * height
#         print('result in original coordinates={}'.format(result))

#         return result

class PredictRequest():

    def __init__(self, headers, content):
        content_length = int(headers['Content-Length'])

        data = content.read(content_length)
        reqJson = json.loads(data)

        imageBytes = base64.b64decode(reqJson['image'])
        buf = np.fromstring(imageBytes, np.uint8)
        self.image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

class RequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args):
        # self.modelServer = ModelServer()
        super().__init__(*args)

    def do_POST(self):
        print('Recevied post')

        if self.path == '/predict':

            req = PredictRequest(self.headers, self.rfile)
            rgb_image = cv2.cvtColor(req.image, cv2.COLOR_BGR2RGB)
            print('starting resize ...')

            x = cv2.resize(req.image, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            print('Done')

            x = torch.from_numpy(x).permute(2, 0, 1)

            xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
            # if torch.cuda.is_available():
            #     xx = xx.cuda()
            print('starting inference ...')
            y = net(xx)
            print('Done')
            detections = y.data
            scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
            result = []
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    score = detections[0, i, j, 0]
                    label_name = labels[i - 1]
                    display_txt = '%s: %.2f' % (label_name, score)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]] # pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    coords = [int(e) for e in coords]
                    result.append(coords)
                    j+=1

            jsonStr = json.dumps(result).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(jsonStr)

    #         ''' From Tensor(1, 2, 2) to Numpy(2, 2)'''
    #         out = out.detach().numpy()

    #         result = {
    #                 'topx': int(out[0,0]), 
    #                 'topy': int(out[0,1]), 
    #                 'bottomx':int(out[1,0]), 
    #                 'bottomy':int(out[1,1]),
    #         }

    #         jsonStr = json.dumps(result).encode()

    #         self.send_response(200)
    #         self.end_headers()
    #         self.wfile.write(jsonStr)

    def do_GET(self):
        print('Received get')

        if self.path == '/status':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'200 ok')
        

def run(server_class=HTTPServer, handler_class=RequestHandler):
        print('Running http server on {}:{}'.format(__ADDRESS__, __PORT__))
        server_address = (__ADDRESS__, __PORT__)
        httpd = server_class(server_address, handler_class)
        httpd.serve_forever()
run()
