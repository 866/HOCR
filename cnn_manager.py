import caffe
import numpy as np
import cv2
import math

FRAME_RESIZE_HEIGHT = 50
class CnnLenetManager:
    def __init__(self, args):
        """
        :param args: key-value dictionary, where key is the name of lenet network
               value - tuple with (prototxt_path, model_path, class_list)
                     class_list - list of characters in restricted order
        :return: object
        """
        self._nets = {}
        self._class_list = {}
        for key, value in args.items():
            self._nets[key] = caffe.Net(value[0], value[1], caffe.TEST)
            self._class_list[key] = value[2]
        caffe.set_mode_cpu()
        self.current_cnn = args.keys()[0]

    def next_mode(self):
        """
        :return:
        """
        keys = self._nets.keys()
        idx = keys.index(self.current_cnn) + 1
        if idx == len(keys):
            idx = 0
        self.current_cnn = keys[idx]

    def get_class(self, img):
        """
        :param img: 28x28x1 image(specially for caffe lenet)
        :return: character
        """
        out = self._nets[self.current_cnn].forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
        max_prob_class = out["prob"][0].argmax()
        return self._class_list[self.current_cnn][max_prob_class]

    def wise_get_class(self, img, threshold = 0.95, out_num = 4):
        """
        :param img: 28x28x1 image(specially for caffe lenet)
        :return: character
        """
        out = self._nets[self.current_cnn].forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
        classes_idx = np.argsort(out["prob"][0])[::-1]
        prob = out["prob"][0][classes_idx[0]]
        if prob < threshold:
            ret = ""
            for i in range(out_num):
                class_idx = classes_idx[i]
                ret += (self._class_list[self.current_cnn][class_idx])
            return ret
        else:
            return self._class_list[self.current_cnn][classes_idx[0]]


class ImageManager:

    def __init__(self, cnn_man, show_mode = "off"):
        """
        :param cnn_man: cnns
        :return:
        """
        self._cnn_man = cnn_man
        self.img = None
        self.blobs = None
        self.show_mode = show_mode

    @staticmethod
    def find_bounds(blob_points):  # finds rectangular frame of the blob
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

    @staticmethod
    def _blob_center_of_mass(blob_points):
        sum_x, sum_y = 0.0, 0.0
        for x, y in blob_points:
            sum_x += x
            sum_y += y
        return sum_x/len(blob_points), sum_y/len(blob_points)

    @staticmethod
    def _validate_point((x, y), img):
        return (x >= 0) and (y >= 0) and (x < img.shape[0]) and (y < img.shape[1] and (img[x,y] != -1))

    def _blobs_filter(self):
        main_blobs = []
        aux_blobs = []
        blobs_average_points = sum([len(blob) for blob in self.blobs])/len(self.blobs)
        if blobs_average_points*0.3 > 20:  # we need at least 20 pts
            for blob in self.blobs:
                dots = len(blob)
                if dots < 10:
                    continue
                if dots >= blobs_average_points*0.35:
                    main_blobs.append(blob)
                elif dots >= blobs_average_points*0.2:
                    aux_blobs.append(blob)
            self.blobs = main_blobs
            for blob in aux_blobs:
                self._find_closest_and_merge(blob)
        else:
            self.blobs = []


    def _find_closest_and_merge(self, blob):
        cm_blob = np.array(self._blob_center_of_mass(blob))
        distance = np.array([])
        for current_blob in self.blobs:
            cm_current_blob = np.array(self._blob_center_of_mass(current_blob))
            diff = cm_current_blob-cm_blob
            out = np.sqrt(diff.dot(diff))
            distance = np.append(distance, [out])
        self.blobs[distance.argmin()] += blob

    def _split_into_blobs(self):
        """
        :return: list of blobs
        """

        if self.img is None:
            raise Exception("Image is not set")

        blobs = []
        img = np.array(self.img[:, :], np.int16)
        step = 0
        way = [(0, 0)]
        # img[way[step]] = -1
        #
        # # Find first zero point
        # while img[way[0]] != 0:
        #     way[0][0] += 1

        while True:
            # Try to find a blob
            blob_point = None

            while (step < len(way)) and (blob_point is None):
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        point = (way[step][0]+dx, way[step][1]+dy)
                        if self._validate_point(point, img):
                            if img[point] != 0:
                                blob_point = point
                                break
                            way.append(point)  # new non-blob point
                            img[point] = -1
                step += 1

            if blob_point is not None:  # found new blob
                blob_points = [blob_point]
                img[blob_point] = -1
                blob_step = 0
                while blob_step < len(blob_points):  # passing blob's points
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            point = (blob_points[blob_step][0]+dx, blob_points[blob_step][1]+dy)
                            if self._validate_point(point, img):
                                if img[point] != 0:
                                    blob_points.append(point)
                                else:
                                    way.append(point)
                                img[point] = -1
                    blob_step += 1
                blobs.append(blob_points)
            else:
                break

        self.blobs = blobs

    def extract_roi_and_process(self, img, roi_region):
        """
        :param img:
        :param roi_region: (left, top, right, bottom) region
        :return:
        """
        self.img = cv2.cvtColor(img[roi_region[1]:roi_region[3], roi_region[0]:roi_region[2]], cv2.COLOR_BGR2GRAY)
        self.img = cv2.blur(self.img, (5, 5))
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)
        self.img = cv2.erode(self.img, (15, 15), iterations=3)
        self.img = cv2.dilate(self.img, (25, 25), iterations=1)
        self.img = cv2.resize(self.img, (int(float(FRAME_RESIZE_HEIGHT)/self.img.shape[0]*self.img.shape[1]),
                                         FRAME_RESIZE_HEIGHT))
        cv2.imshow("Processed roi", self.img)


    def set_img(self, img):
        """
        :param img: assigns to inner member img
        :return:
        """
        self.img = img

    def find_blobs(self, filtering=True):
        """
        :param filter: filtrate and merge(true) or not(false)
        :return:
        """
        self._split_into_blobs()

        if filtering:
            self._blobs_filter()

        print str(len(self.blobs)) + " blobs are found."

    def reproduce_blob(self, blob_points):
        top, left, bottom, right = self.find_bounds(blob_points)
        center_x, center_y = self._blob_center_of_mass(blob_points)
        if bottom-top > right - left:
            max_side = bottom-top
        else:
            max_side = right-left
        tmp_img = np.zeros((max_side*1.4, max_side*1.4, 1), np.uint8)
        dx = math.floor(0.7*max_side-center_x)
        dy = math.floor(0.7*max_side-center_y)
        for x, y in blob_points:
            if self._validate_point((x+dx, y+dy), tmp_img):
                tmp_img[x+dx, y+dy, 0] = 255
        tmp_img = caffe.io.resize_image(tmp_img, (28, 28))
        return tmp_img

    def classify_blobs(self):
        res = []
        if self.blobs is None:
            raise Exception("There are no blobs. Maybe you should run find_blobs first")
        for blob in self.blobs:
            img = np.array((28, 28, 1), np.uint8)
            img = self.reproduce_blob(blob)
            label = self._cnn_man.wise_get_class(img)
            res.append(label)

            if self.show_mode == "on":
                import matplotlib.pyplot as plt
                plt.imshow(img[:, :, 0])
                plt.title(label)
                plt.show()
        return res





