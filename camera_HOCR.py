__author__ = 'victor'

import cnn_manager as cnn
import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)
MAX_CAM_NUM = 2

small_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
cap_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
digits = ['0','1','2','3','4','5','6','7','8','9']

lenets = { "digits" : ("/home/victor/Programming/caffe/examples/CAFFE_CNN/mnist/lenet.prototxt",
                       "/home/victor/Programming/caffe/examples/CAFFE_CNN/mnist/lenet_iter_30000.caffemodel",
                       digits),
           "small letters": ("/home/victor/Programming/caffe/examples/CAFFE_CNN/small_letters_experimental/lenet.prototxt",
                             "/home/victor/Programming/caffe/examples/CAFFE_CNN/small_letters_experimental/lenet_iter_20000.caffemodel",
                             small_letters),
           "capital letters": ("/home/victor/Programming/caffe/examples/CAFFE_CNN/cap_letters_CNN/lenet.prototxt",
                               "/home/victor/Programming/caffe/examples/CAFFE_CNN/cap_letters_CNN/lenet_iter_40000.caffemodel",
                               cap_letters)

}

cams = [cv2.VideoCapture(i) for i in range(MAX_CAM_NUM)]
cam_num = 1
cnn_man = cnn.CnnLenetManager(lenets)
img_man = cnn.ImageManager(cnn_man)
cap = cams[cam_num]
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
roi = (int(width*0.1), int(height*0.15), int(width*0.9), int(height*0.35))
font = cv2.FONT_HERSHEY_SIMPLEX

while cap:

    ret, frame = cap.read()
    if frame is None:
        continue

    output_img = frame.copy()
    cv2.rectangle(output_img, roi[0:2], roi[2:4], GREEN, 2)
    cv2.putText(output_img,"Current set: " + cnn_man.current_cnn, (10, 40), font, 0.4, GREEN, 1)
    cv2.putText(output_img,"Show mode: " + img_man.show_mode, (10, 60), font, 0.4, GREEN, 1)
    cv2.imshow("HOCR.  Cam " + str(cam_num), output_img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord(' '):
        cv2.putText(output_img,"Recognizing roi...",(int(height*.4), int(width*.4)), font, 1, GREEN, 1)
        cv2.imshow("HOCR.  Cam " + str(cam_num), output_img)
        cv2.waitKey(10)
        img_man.extract_roi_and_process(frame, roi)
        img_man.find_blobs()
        print "\nRecognized characters:"
        print(img_man.classify_blobs())

    if key & 0xFF == ord('c'):
        cv2.destroyAllWindows()

        cam_num += 1
        if cam_num == 2:
            cam_num = 0

        cap = cams[cam_num]
        height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        roi = (int(width*0.1), int(height*0.15), int(width*0.9), int(height*0.35))

    if key & 0xFF == ord('s'):  # switch to next mode
        cnn_man.next_mode()

    if key & 0xFF == ord('h'):  # switch show mode
        if img_man.show_mode == "on":
            img_man.show_mode = "off"
        else:
            img_man.show_mode = "on"

cap.release()
cv2.destroyAllWindows()
