__author__ = 'victor'

import sys
sys.path.append("/home/victor/Programming/caffe/python")
import caffe
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import time


def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def validate_point((x, y), img):
    return (x >= 0) and (y >= 0) and (x < img.shape[0]) and (y < img.shape[1] and (img[x,y] != -1))


def find_blobs(img_orig):
    blobs = []
    img = np.array(img_orig[:, :], np.int16)
    step = 0
    way = [(0, 0)] # assumes that first point is not a blob(can be mistake but usually not)
                # rewrite it
    img[way[step]] = -1
    while True:
        # Try to find a blob
        blob_point = None

        while (step < len(way)) and (blob_point is None):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    point = (way[step][0]+dx, way[step][1]+dy)
                    if validate_point(point, img):
                        if img[point] == 255:
                            blob_point = point
                            break
                        way.append(point) # new non-blob point
                        img[point] = -1
            step += 1

        if blob_point is not None: # found new blob
            blob_points = [blob_point]
            img[point] = -1
            blob_step = 0
            while blob_step < len(blob_points): #passing blob's points
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        point = (blob_points[blob_step][0]+dx, blob_points[blob_step][1]+dy)
                        if validate_point(point, img):
                            if img[point] == 255:
                                blob_points.append(point)
                            else:
                                way.append(point)
                            img[point] = -1
                blob_step += 1
            blobs.append(blob_points)
        else:
            break

    return blobs


def blob_center_of_mass(blob_points):
    sum_x, sum_y = 0.0, 0.0
    for x, y in blob_points:
        sum_x += x
        sum_y += y
    return sum_x/len(blob_points), sum_y/len(blob_points)

def reproduce_blob(blob_points, img):
    center_x, center_y = blob_center_of_mass(blob_points)
    dx = int(img.shape[0]/2.0-center_x)
    dy = int(img.shape[1]/2.0-center_y)
    for x, y in blob_points:
        if validate_point((x+dx, y+dy), img):
            img[x+dx, y+dy] = 255

def binary(caffe_img, threshold = 0):
    height, width = caffe_img.shape[:2]
    ret_img = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            if caffe_img[i, j, 0] > threshold:
                ret_img[i, j, 0] = 255
            else:
                ret_img[i, j, 0] = 0
    return ret_img


image_file = sys.argv[1]
print "Recognize ", image_file

orig_img = caffe.io.load_image(image_file, color=False)
# cv2.namedWindow("Original image")
# cv2.imshow("Original image", img)
img = orig_img

height, width = img.shape[:2]
resized_width = int(28.0/float(height)*width)
print "Resize: "+str(height)+"x"+str(width)+" to 28x"+str(resized_width)
img = caffe.io.resize_image(img, (28, resized_width))

print "Convert image to network type:"
img = img*(255/img.max())
img = binary(img, 20)

blobs = find_blobs(img[:, :, 0])
print "Found " + str(len(blobs)) + " blobs"
# for blob in blobs:
#     img = np.zeros((28, 28, 1))
#     reproduce_blob(blob, img)
#     plt.imshow(img[:,:,0])
#     plt.show()


print "Load caffe network"
MODEL_FILE = 'lenet.prototxt'
PRETRAINED = 'lenet_iter_15000.caffemodel'
net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_mode_cpu()


print "Obtain result"
result_list = []
current_pos = 0
start = time.time()
# while current_pos+28 <= resized_width:
#     cut = img[:, current_pos:current_pos+28,:]
#     #plt.imshow(cut[:,:,0])
#     #plt.show()
#     out = net.forward_all(data=np.asarray([cut.transpose(2,0,1)]))
#     result_list.append(out['prob'][0].argmax())
#     current_pos += step
for blob in blobs:
    blob_img = np.zeros((28, 28, 1))
    reproduce_blob(blob, blob_img)
    # plt.imshow(blob_img[:,:,0])
    # plt.show()
    out = net.forward_all(data=np.asarray([blob_img.transpose(2,0,1)]))
    result_list.append(out['prob'][0].argmax())

end = time.time()
print result_list
print "Time elapsed: " + str(end - start) + " s"
