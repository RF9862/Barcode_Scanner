from pyzbar import pyzbar
import cv2, gc
import pytesseract
from pytesseract import Output
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import os, sys

def extractBarcode(image):
    barcodes = pyzbar.decode(image)
    barcode_data = []
    qrcode_data = []
    barcodeOri = None
    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        if barcodeOri is None: barcodeOri = barcode.orientation
        
        if barcodeType == "QRCODE":
            qrcode_data.append([barcodeType, barcodeData, [x,y,w,h]])
        else:
            barcode_data.append([barcodeType, barcodeData, [x,y,w,h]])
        # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    del image
    return barcode_data, qrcode_data, barcodeOri

def code_loc(ref_y, data, mid_coor):
    if len(ref_y) == 0:
        code1, code2 = 0, 0
    elif len(ref_y) == 1:
        if ref_y[0] > mid_coor/2: code1, code2 = data[0], 0
        else: code1, code2 = 0, data[0]
    else:
        code1 = data[ref_y.index(max(ref_y))]
        code2 = data[ref_y.index(min(ref_y))]
    return code1, code2

def barcode_ang(data, barcodeOri, img_w, img_h, image):

    for i in range(len(data)):
    #############################################
        try:
            x, y, w, h = data[i][2]
        except:
            pass
        if barcodeOri == "RIGHT":
            if image is not False and i == 0: image = np.rot90(image,1) # 90 deg
            data[i][2] = [y, img_w-(x+w), h, w]
        elif barcodeOri == "DOWN":
            if image is not False and i == 0: image = np.rot90(image,2) # 180 deg
            data[i][2] = [img_w-(x+w), img_h-(y+h), w, h]      
        elif barcodeOri == "LEFT":
            if image is not False and i == 0: image = np.rot90(image,3) # 270 deg
            data[i][2] = [img_h-(y+h), x, h, w]  
    return data, image
def temp_contour(temp_hand, tk, img_h, img_w, erod_size=0):
    new_hand_img = np.zeros_like(temp_hand)
    contours, _ = cv2.findContours(temp_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < img_h*0.9 and w < img_h*0.9 and ((h >30+erod_size and w > 6+erod_size) or (w>20+erod_size and h>10+erod_size)):
            if w > 20: new_hand_img[y:y+h, x+tk:x+w-tk] =255       
            else: new_hand_img[y:y+h, x:x+w] =255      
            # cv2.rectangle(hand_img_cpy, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            
    temp_ind = []            
    contours, _ = cv2.findContours(new_hand_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < img_h*0.9 and w < img_h*0.9:# and (h >25 or w > 30-tk*2):
            # cv2.rectangle(hand_img_cpy, (x-tk, y), (x + w+2*tk, y + h), (0, 0, 255), 2)
            temp_ind.append([x-tk, y, w+2*tk, h])  
    temp_ind.sort()  
    # for i in range(len(temp_ind)-1):

    #     if temp_ind[i][0]+temp_ind[i][2] < temp_ind[i+1][0] -2*tk:
    #         temp_ind[i][2], temp_ind[i+1][0] = temp_ind[i][2]+2*tk, temp_ind[i+1][0] - tk
    #         if i == 0: 
    #             temp_ind[i][0] = temp_ind[i][0]-tk
    #         elif i == len(temp_ind)-1: 
    #             temp_ind[i+1][2] = temp_ind[i+1][2]+2*tk

    return temp_ind
  
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# @profile
def main_parse(filename, template_path, temp_imgs):
    '''
    filename: image full name to be processed.
    template_path: path of template.json file
    temp_imgs: path to save cropped images of total value
    '''
    image = cv2.imread(filename)
    img_h, img_w, _ = image.shape
    barcode_data, qrcode_data, barcodeOri = extractBarcode(image)

    qrcode_data, image = barcode_ang(qrcode_data, barcodeOri, img_w, img_h, image)
    barcode_data, _ = barcode_ang(barcode_data, barcodeOri, img_w, img_h, False)
    #####################################
    qr_y = [v[2][1] for v in qrcode_data]
    bar_y = [v[2][0] for v in barcode_data]
    
    img_h, img_w, _ = image.shape
    QR2, QR1 = code_loc(qr_y, qrcode_data, img_h)
    Bar2, Bar1 = code_loc(bar_y, barcode_data, img_h)

    if Bar1 == 0 and Bar2 == 0 and QR1 == 0 and QR2 == 0:
        raise Exception("Input is wrong...")
    ### finding total value ###
    total_img_w, total_img_h = 500, 240
    if QR2 != 0:
        total_img_x0 = QR2[2][0] + QR2[2][2]+280
        total_img_y0 = int(QR2[2][1]+QR2[2][3]/2)-115
        total_img = image[total_img_y0:total_img_y0+total_img_h, total_img_x0:total_img_x0+total_img_w].copy()
    else:
        json_file = resource_path(os.path.join(template_path, 'template.json'))
        with open(json_file, "r") as f:
            xyxy = json.load(f)
        total_img = image[xyxy["y0"]:xyxy["y1"], xyxy["x0"]:xyxy["x1"]].copy()
    nm = filename.split('/')[-1]
    flag = cv2.imwrite(f"{temp_imgs}/{nm}", total_img)
    if not flag: cv2.imwrite(f"{temp_imgs}/{nm}", np.ones([100,100,3],dtype=np.uint8)*255)

    del (image, total_img)
    # return 'QR1', 'QR2', 'Bar1', 'Bar2', 'total_value', 'min_prob', 'consider_img'
    if QR1 != 0 and QR2 != 0: return [QR1[1], QR2[1]]
    elif QR1 != 0: return [QR1[1], QR1[1]]
    elif QR2 != 0: return [QR2[1], QR2[1]]
    elif Bar1 != 0: return [Bar1[1], Bar1[1]]
    elif Bar2 != 0: return [Bar2[1], Bar2[1]]
    else: return ['Not Recognized', 'Not Recognized']
    
    # QR1 = 'Not Recognized' if QR1 == 0 else QR1[1]
    # QR2 = 'Not Recognized' if QR2 == 0 else QR2[1]
    # Bar1 = 'Not Recognized' if Bar1 == 0 else Bar1[1]
    # Bar2 = 'Not Recognized' if Bar2 == 0 else Bar2[1]
    
    # return QR1, QR2, Bar1, Bar2


    