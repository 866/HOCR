__author__ = 'victor'

import sys
import caffe
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb
import time

def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


image_file = sys.argv[1]
split_num = int(sys.argv[2])
print "Recognize ", image_file, " splitted into ", split_num
orig_img = caffe.io.load_image(image_file, color=False)
#cv2.namedWindow("Original image")
#cv2.imshow("Original image", img)
img=orig_img

height, width = img.shape[:2]
resized_width = int(28.0/float(height)*width)
print "Resize: "+str(height)+"x"+str(width)+" to 28x"+str(resized_width)
img = caffe.io.resize_image(img, (28, resized_width))



print "Load caffe network"
MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'lenet_iter_10000.caffemodel'
net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_mode_cpu()

print "Convert image to network type:"
img = img*255
img = img.astype(np.uint8)


print "Obtain result"
result_list = []
step = int(resized_width/split_num)
current_pos = 0
start = time.time()
while(current_pos+28 <= resized_width):
    cut = img[:, current_pos:current_pos+28,:]

    out = net.forward_all(data=np.asarray([cut.transpose(2,0,1)]))
    result_list.append(out['prob'][0].argmax())
    current_pos += step
end = time.time()
print result_list
print "Time elapsed: " + str(end - start) + " s"