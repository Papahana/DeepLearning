import cv2
import numpy as np
import sys
import pytesseract  # Also need to install Tesseract -> https://github.com/UB-Mannheim/tesseract/wiki
import tensorflow as tf
import os
import warnings

from dlgpu_assistant.dl_tools import DlAssistant
from keras.utils import load_img, img_to_array
from keras.models import load_model

DlAssistant().tf_warnings()
warnings.filterwarnings('ignore')
tf.compat.v1.disable_eager_execution()

# Directory where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = "P:\Tesseract\\tesseract"

# Image Directory - Change it with your own
dir_license = "License_Plates/Cars1.png"

image_data = cv2.imread(dir_license)
gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
bt_image = cv2.bilateralFilter(gray_image, 10, 75, 75)  # Default: (gray_image, 11, 17, 17)
edged = cv2.Canny(bt_image, 50, 150)
cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

plate = None
for c in cnts:
    epsilon = cv2.arcLength(c, True) * 0.02  # Default 0.018
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        plate = image_data[y:y + h, x:x + w]
        print("License Plate Found")
        break

if plate is None:
    print("License Plate not Found. Killing process.")
    sys.exit()


def find_contours(dimensions, img, plate):
    cntrs, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    x_contour = []
    result = []
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)

        if lower_width < w < upper_width and lower_height < h < upper_height:
            x_contour.append(x)  

            char = img[y:y + h, x:x + w]
            char = cv2.resize(char, (128, 128))  # Default (20, 40)

            plate = cv2.rectangle(plate, (x, y), (w + x, y + h), (255, 0, 0), 2)

            char = cv2.subtract(255, char)
            result.append(char)

    indices = sorted(range(len(x_contour)), key=lambda k: x_contour[k])
    result_copy = []
    for idx in indices:
        result_copy.append(result[idx])  
    result = np.array(result_copy)

    return result, plate


def segment_characters(image):
    plate = cv2.resize(image, (333, 75))
    gray_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, img_binary_lp = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    eroded_img = cv2.erode(img_binary_lp, (3, 3))
    dilated_img = cv2.dilate(eroded_img, (3, 3))

    # Y.shape is (n, m) ---- n for rows and m for columns
    LP_WIDTH = dilated_img.shape[0]
    LP_HEIGHT = dilated_img.shape[1]
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2 * LP_HEIGHT/3]

    char_list, plate = find_contours(dimensions, dilated_img, plate)

    return char_list, plate


char, plate = segment_characters(plate)


def prediction(char, plate):
    model = load_model('Model/CNN_model.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    plate_prediction = []
    for i in range(len(char)):
        # char[i] = cv2.bitwise_not(char[i])
        cv2.imwrite("letter.png", char[i])
        # cv2.imshow("Character", char[i])
        # cv2.waitKey(0)

        img = load_img("letter.png", target_size=(128, 128))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        predict_x = model.predict(images, verbose=0)
        classes = np.argmax(predict_x, axis=1)

        char_real = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        plate_prediction.append(char_real[classes[0]])
        os.remove("letter.png")

    cv2.imwrite("last_prediction.png", plate)
    cv2.imshow("Characters", plate)
    print("Prediction: " + "".join(plate_prediction) + "\n")
    print("Press Enter to finish")
    cv2.waitKey(0)


prediction(char, plate)
