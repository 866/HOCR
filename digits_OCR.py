__author__ = 'victor'

import sys
sys.path.append("/home/victor/Programming/caffe-master/python")
import caffe
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math


def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def validate_point((x, y), img):
    return (x >= 0) and (y >= 0) and (x < img.shape[0]) and (y < img.shape[1] and (img[x,y] != -1))


def find_closest_and_merge(blob, blobs):
    cm_blob = np.array(blob_center_of_mass(blob))
    distance = np.array([])
    for current_blob in blobs:
        cm_current_blob = np.array(blob_center_of_mass(current_blob))
        diff = cm_current_blob-cm_blob
        out=np.sqrt(diff.dot(diff))
        distance = np.append(distance, [out])
    blobs[distance.argmin()] += blob



def blobs_filter(blobs):
    main_blobs = []
    aux_blobs = []
    blobs_average_points = sum([len(blob) for blob in blobs])/len(blobs)
    print "Blobs_len: ", len(blobs)
    if (blobs_average_points*0.3 > 20): # we need at least 20 pts
        for blob in blobs:
            if len(blob) >= blobs_average_points*0.35:
                main_blobs.append(blob)
            elif len(blob) >= blobs_average_points*0.03:
                aux_blobs.append(blob)
        for blob in aux_blobs:
            find_closest_and_merge(blob, main_blobs)
    return main_blobs


def find_bounds(blob_points): # finds rectangular bound of the blob
    top, left = blob_points[0]
    bottom, right = blob_points[0]
    for x, y in blob_points:
        if top > x:
            top = x
        elif bottom < x:
            bottom = x
        if left > y:
            left = y
        elif right < y:
            right = y
    return top, left, bottom, right


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
                        if img[point] > 0:
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
                            if img[point] > 0:
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


def reproduce_blob(blob_points):
    top, left, bottom, right = find_bounds(blob_points)
    center_x, center_y = blob_center_of_mass(blob_points)
    if bottom-top > right - left:
        max_side = bottom-top
    else:
        max_side = right-left
    tmp_img = np.zeros((max_side*1.4, max_side*1.4, 1), np.uint8)
    dx = math.floor(0.7*max_side-center_x)
    dy = math.floor(0.7*max_side-center_y)
    for x, y in blob_points:
        if validate_point((x+dx, y+dy), tmp_img):
            tmp_img[x+dx, y+dy, 0] = 255
    tmp_img = caffe.io.resize_image(tmp_img, (28, 28))
    return tmp_img


def binary(caffe_img, threshold=0):
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
img = orig_img

print "Convert image to network type:"
img = img*(255/img.max())

blobs = blobs_filter(find_blobs(img[:, :, 0]))
print "Found " + str(len(blobs)) + " blobs"

print "Load caffe network"
MODEL_FILE = '/home/victor/Programming/caffe-master/examples/small_letters_experimental/lenet.prototxt'
PRETRAINED = '/home/victor/Programming/caffe-master/examples/small_letters_experimental/lenet_iter_20000.caffemodel'
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
print len(blobs)
for blob in blobs:
    blob_img = reproduce_blob(blob)
    if len(sys.argv) > 2:
        plt.imshow(blob_img[:, :, 0])
        plt.show()
    out = net.forward_all(data=np.asarray([blob_img.transpose(2,0,1)]))
    result_list.append(chars[out['prob'][0].argmax()])


end = time.time()
print result_list
print "Time elapsed: " + str(end - start) + " s"
