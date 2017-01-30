import sys
sys.path.insert(0, '/home/deep/rahim/caffe-master/python')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os
import os.path
import sys
sys.path.insert(0, '/home/deep/rahim/caffe-master/python')



rootdir = '/home/deep/rahim/Dataset/Eyediap/Data-Gaze/frames/EYEDIAP15/15_A_DS_M/Eyes/Left'

#load the model
net = caffe.Net('/home/deep/rahim/PGM/Final/feature/deep/Lenet/deploy.prototxt',
                '/home/deep/rahim/PGM/Final/feature/deep/Lenet/snapshot_iter_40000.caffemodel',
                caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)



def classifyOne(inputPng):
    IMAGE_FILE = os.path.join(rootdir,inputPng)
    input_image = caffe.io.load_image(IMAGE_FILE)
    plt.imshow(input_image)
    #input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
    #net.blobs['data'].data[...] = np.asarray(transformer.preprocess('data', in_) for in_ in input_oversampled)
    net.blobs['data'].data[...] = transformer.preprocess('data', input_image)    
    net.blobs['data'].reshape(1,3,40,40)
    caffe.set_mode_gpu()
    out = net.forward()
    #print '%f %f %f %f %f %f %f %f %f %f %f %f' %(out ['predict01'],out ['predict02'],out ['predict03'],out ['predict04'],out ['predict05'],out ['predict06'],out ['predict07'],out ['predict08'],out ['predict09'],out ['predict10'],out ['predict11'],out ['predict12'])
    print '%s' %(out ['ip2'])
num = 1
for fileList in os.walk(rootdir):
    #print len(fileList)
    filenames = fileList[2]
    #print len(filenames)
    for one in filenames:
        classifyOne(one)

#print predicted labels
#labels = np.loadtxt("/home/deep/rahim/FinetuneVGG-Eyediap/16Layer/scenario8/labels.txt", str, delimiter='\t')
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#print labels[top_k]

