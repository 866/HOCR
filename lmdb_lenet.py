__author__ = 'victor'

import sys
import caffe
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb
import matplotlib.image as mpimg
caffe_root = '../'

MODEL_FILE = '/home/victor/Programming/caffe-master/examples/small_letters/lenet.prototxt'
PRETRAINED = '/home/victor/Programming/caffe-master/examples/small_letters/lenet_iter_5000.caffemodel'
#
chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_mode_cpu()
# Test self-made image

db_path = '/home/victor/Programming/caffe-master/examples/small_letters/small_letters_test_lmdb'
lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0
for key, value in lmdb_cursor:
    print "Count:"
    print count
    count = count + 1
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = np.zeros((datum.channels, datum.height, datum.width))
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]))
    print image
    image = np.transpose(image, (1, 2, 0))
    predicted_label = out['prob'][0].argmax(axis=0)
    print out['prob']
    if label == predicted_label:
        correct = correct + 1
    print("Label is class " + chars[label] + ", predicted class is " + chars[predicted_label])
    mpimg.imsave(str(label)+"_"+str(count), image[:,:,0], cmap=plt.get_cmap("Greys"))
    plt.imshow(image[:,:,0])
    plt.show()


print(str(correct) + " out of " + str(count) + " were classified correctly")

