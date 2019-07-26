# -*- coding: utf-8 -*-
# python2.7
import cv2 as cv 
import numpy as np
import math
import caffe
import time
from config_reader import config_reader
class Keypoint_Inference(object):
    def __init__(self):
        params, model = config_reader()
        self.boxsize = model['boxsize']
        self.center = self.boxsize / 2
        self.npart = model['np']
        self.target_height = float(model['target_height'])

        # caffe.reset_all()
        if params['use_gpu']:
            caffe.set_mode_gpu()
            caffe.set_device(params['GPUdeviceNumber'])
        else:
            caffe.set_mode_cpu()
        self.hand_net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
        # self.hand_net = caffe.Net(model['deployFile'], caffe.TEST)
        # self.hand_net.copy_from(model['caffemodel'])
        self.hand_net.forward()
        self.prediction = np.zeros((self.npart, 2))
        self.gaussian_map = np.zeros((self.boxsize, self.boxsize))
        for y_p in range(self.boxsize):
            for x_p in range(self.boxsize):
                dist_sq = (x_p - self.center) * (x_p - self.center) + (y_p - self.center) * (y_p - self.center)
                exponent = dist_sq / 2.0 / model['sigma'] / model['sigma']
                self.gaussian_map[y_p, x_p] = math.exp(-exponent)

    def predict(self, oriImg, hand_box):
        '''
        oriImg: image to test, now must be bgr 3-channel image
        hand_box: [xmin, xmax, ymin, ymax], hand bbox provided by user or by hand detector
        '''
        # scale
        real_scale = self.target_height / (hand_box[3] - hand_box[2])
        imageToTest = cv.resize(oriImg, (0,0), fx=real_scale, fy=real_scale, interpolation=cv.INTER_CUBIC)
        hand_center = [int(round((hand_box[0]+hand_box[1])/2.0*real_scale)), int(round((hand_box[2]+hand_box[3])/2.0*real_scale))]
        # scaling done
        
        # crop
        hand_image = np.ones((self.boxsize, self.boxsize, 3)) * 128

        top = hand_center[1] - self.center
        hand_top = 0
        if top < 0:
            hand_top = -top
            top = 0
        
        left = hand_center[0] - self.center
        hand_left = 0
        if left < 0:
            hand_left = -left
            left = 0

        bottom = hand_center[1] + self.center
        hand_bottom = self.boxsize
        if bottom > imageToTest.shape[0]:
            hand_bottom -= bottom - imageToTest.shape[0]
            bottom = imageToTest.shape[0]

        right = hand_center[0] + self.center
        hand_right = self.boxsize
        if right > imageToTest.shape[1]:
            hand_right -= right - imageToTest.shape[1]
            right = imageToTest.shape[1]

        hand_image[hand_top:hand_bottom, hand_left:hand_right] = imageToTest[top:bottom, left:right]
        # cropping done
        cv.imwrite('hand_image.jpg', hand_image)

        input_4channels = np.ones((self.boxsize, self.boxsize, 4))
        input_4channels[:,:,0:3] = hand_image / 256.0 - 0.5 # normalize to [-0.5, 0.5]
        input_4channels[:,:,3] = self.gaussian_map
        self.hand_net.blobs['data'].data[...] = np.transpose(np.float32(input_4channels[:,:,:,np.newaxis]), (3,2,0,1))
        start_time = time.time()
        output_blob = self.hand_net.forward()['Mconv5_stage3']
        print('net took %.2f ms.' % (1000 * (time.time() - start_time)))

        for part in range(self.npart):
            part_map = output_blob[0, part, :, :]
            part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
            self.prediction[part,:] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)

        self.prediction[:,0] = (hand_center[1] - self.center + self.prediction[:,0]) / real_scale
        self.prediction[:,1] = (hand_center[0] - self.center + self.prediction[:,1]) / real_scale

        return self.prediction

class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Dense(3, activation='relu'),
            nn.Dense(4, activation='relu'))
        self.dense = nn.Dense(5)
    
    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)

    # net = MixMLP()
    # net
    # net.initialize()
    # x = nd.random.uniform(shape=(2,2))
    # net(x)
    # net.blk[1].weight.data()

def main():
    keypoint_detector = Keypoint_Inference()

    test_image = 'Image1529571920899.jpg'
    hand_box = [321, 525, 86, 366] # [xmin xmax ymin ymax]
    oriImg = cv.imread(test_image, cv.IMREAD_COLOR)
    prediction = keypoint_detector.predict(oriImg, hand_box) # shape = (5, 2); dtype = np.float64
    for part in range(keypoint_detector.npart):
        cv.circle(oriImg, (int(round(prediction[part, 1])), int(round(prediction[part, 0]))), 3, (255, 255, 0), -1)
    cv.imwrite('detected.jpg', oriImg)

    test_image = 'Image1529576385972.jpg'
    hand_box = [440, 650, 244, 484]
    oriImg = cv.imread(test_image, cv.IMREAD_COLOR)
    prediction = keypoint_detector.predict(oriImg, hand_box)
    for part in range(keypoint_detector.npart):
        cv.circle(oriImg, (int(round(prediction[part, 1])), int(round(prediction[part, 0]))), 3, (255, 255, 0), -1)
    cv.imwrite('detected_.jpg', oriImg)

if __name__ == '__main__':
    main()