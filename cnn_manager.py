import caffe
import numpy as np
import cv2

class CnnLenetManager:
    def __init__(self, **args):
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
            self._class_list = value[3]
        caffe.set_mode_cpu()
        self.current_cnn = args.keys()[0]

    def next_mode(self):
        """
        :return: name of the next mode in string
        """
        keys = self._nets.keys()
        idx = keys.index(self.current_cnn) + 1
        if idx == len(keys):
            idx = 0
        return keys[idx]

    def get_class(self, img):
        """
        :param img: 28x28x1 image(specially for caffe lenet)
        :return: character
        """
        out = self._nets[self.current_cnn].forward_all(data=np.asarray([img.transpose(2, 0, 1)]))
        return self._class_list[self.current_cnn][out["prob"][0].argmax()]


class ImageManager:

    def __init__(self, cnn_man):
        """
        :param cnn_man: cnns
        :return:
        """
        self._cnn_man = cnn_man
        self.img = None
        self.blobs = None

    @staticmethod
    def find_bounds(self, blob_points):  # finds rectangular frame of the blob
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
    def _blob_center_of_mass(self, blob_points):
        sum_x, sum_y = 0.0, 0.0
        for x, y in blob_points:
            sum_x += x
            sum_y += y
        return sum_x/len(blob_points), sum_y/len(blob_points)

    @staticmethod
    def _validate_point((x, y), img):
        return (x >= 0) and (y >= 0) and (x < img.shape[0]) and (y < img.shape[1] and (img[x,y] != -1))

    @staticmethod
    def _blobs_filter(self, blobs):
        main_blobs = []
        aux_blobs = []
        blobs_average_points = sum([len(blob) for blob in blobs])/len(blobs)
        if blobs_average_points*0.3 > 20:  # we need at least 20 pts
            for blob in blobs:
                if len(blob) >= blobs_average_points*0.35:
                    main_blobs.append(blob)
                elif len(blob) >= blobs_average_points*0.03:
                    aux_blobs.append(blob)
            for blob in aux_blobs:
                self._find_closest_and_merge(blob, main_blobs)
        return main_blobs

    @staticmethod
    def _find_closest_and_merge(self, blob, blobs):
        cm_blob = np.array(self._blob_center_of_mass(blob))
        distance = np.array([])
        for current_blob in blobs:
            cm_current_blob = np.array(self._blob_center_of_mass(current_blob))
            diff = cm_current_blob-cm_blob
            out=np.sqrt(diff.dot(diff))
            distance = np.append(distance, [out])
        blobs[distance.argmin()] += blob

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
        img[way[step]] = -1

        # Find first zero point
        while img[way[0]] != 0:
            way[0][0] += 1

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

        return blobs

    def extract_roi_and_process(self, img, roi_region):
        """
        :param img:
        :param roi_region: (left, top, right, bottom) region
        :return:
        """
        self.img = cv2.cvtColor(img[roi_region[0]:roi_region[2], roi_region[1]:roi_region[3]], cv2.COLOR_BGR2GRAY)
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

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
        if filtering:
            self.blobs = self._blobs_filter(self._split_into_blobs())
        else:
            self.blobs = self._split_into_blobs()

    def classify_blobs(self):
        res = []
        if self.blobs is None:
            raise Exception("There are no blobs. Maybe you should run find_blobs first")
        for blob in self.blobs:
            self._cnn_man.get_class(rep)


