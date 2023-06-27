from pyzbar import pyzbar
import cv2, gc
import pytesseract
from pytesseract import Output
import numpy as np
from keras.models import load_model
from skimage.metrics import structural_similarity as ssim
import json
import os, sys

def file(image):
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

def getting_textdata(img, conf):
    '''
    img: soucr image to process.
    conf: tesseract conf (--psm xx)
    '''
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config=conf)
    text_ori = d['text']
    left_coor, top_coor, wid, hei, conf = d['left'], d['top'], d['width'], d['height'], d['conf']        
    ### removing None element from text ###
    text, left, top, w, h, accu, xc, yc= [], [], [], [], [], [], [], []
    for cnt, te in enumerate(text_ori):
        if te.strip() != '' and wid[cnt] > 10 and hei[cnt] > 10:
            text.append(te)
            left.append(int((left_coor[cnt])))
            top.append(int(top_coor[cnt]))
            w.append(int(wid[cnt]))
            h.append(int(hei[cnt]))
            accu.append(conf[cnt])    
            xc.append(int((left_coor[cnt]+wid[cnt]/2)))
            yc.append(int((top_coor[cnt]+hei[cnt]/2)))
    return text, left, top, w, h, accu, xc, yc
def subset(set, lim, loc):
    '''
    set: one or multi list or array, lim: size, loc:location(small, medi, large)
    This function reconstructs set according to size of lim in location of loc.
    '''
    cnt, len_set = 0, len(set)        
    v_coor_y1, index_ = [], []
    pop = []
    for i in range(len_set):
        if i < len_set-1:
            try:
                condition = set[i+1][0] - set[i][0]
            except:
                condition = set[i+1] - set[i]
            if condition < lim:
                cnt = cnt + 1
                pop.append(set[i])
            else:
                cnt = cnt + 1
                pop.append(set[i])
                pop = np.asarray(pop)
                try:
                    if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                    else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                except:
                    if loc == "small": v_coor_y1.append(min(pop))
                    elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                    else: v_coor_y1.append(max(pop))  
                index_.append(cnt)
                cnt = 0
                pop = []
        else:
            cnt += 1
            pop.append(set[i])
            pop = np.asarray(pop)
            try:
                if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
            except:
                if loc == "small": v_coor_y1.append(min(pop))
                elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                else: v_coor_y1.append(max(pop))                    
            index_.append(cnt)

    return v_coor_y1, index_
        
def lines_extraction(img, setting):
    '''
    1. Convert source image into inv grayscal.
    2. Strenthen color contrast of image.
    3. Considering broken lines or faint images, run cv2.filter2D()
    4. Get all lines
    5. Extract lines satisfied some conditions 
    '''
    # Convert source image into inv grayscal.
    # img_gry = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Strenthen color contrast of image.
    # try:
    #     img_gry = (img_gry>50)*(np.zeros_like(img_gry)+255) + (img_gry<=50)*np.zeros_like(img_gry)
    # except:
    #     img_gry = (img_gry>50)*(np.zeros_like(img_gry)+255) + (img_gry<=50)*np.zeros_like(img_gry)
    kernel = np.ones((1,2),np.float32)/2
    # Considering broken lines or faint images, run cv2.filter2D()
    new = cv2.filter2D(img,-1,kernel)
    # th, bin_img = cv2.threshold(img_gry, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img_cny = cv2.Canny(img_gry, 50, 200)
    # Get all lines
    fld = cv2.ximgproc.createFastLineDetector()
    lns = fld.detect(new) 
    # xxx = fld.drawSegments(img.copy(), lns) # for drawing
    img_cpy = img.copy()
    lim, hor_con = setting[0], setting[1]
    lines = []
    x_min, x_max = 10000, 0
    # Extract lines satisfied some conditions 
    if lns is None: pass
    else:                 
        for ln in lns:
            x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])                   
            if abs(y1-y2) < lim and abs(x1-x2) > hor_con:
                if x_min > min(x1, x2): x_min = min(x1, x2)
                if x_max < max(x1, x2): x_max = max(x1, x2)
                lines.append(int(y1/2+y2/2))
                
                # lines.append([int(y1/2+y2/2), min(x1, x2), max(x1, x2)])
                # cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=5)      
        lines.sort()         

    return x_min, x_max, lines

def border_set(img_, coor, tk, color):
    '''
    coor: [x0, x1, y0, y1] - this denotes border locations.
    tk: border thickness, color: border color.
    '''
    img = img_.copy()
    if coor[0] != None:
        img[:, coor[0]:coor[0]+tk] = color # left vertical
    if coor[1] != None:
        img[:, coor[1]-tk:coor[1]] = color # right vertical
    if coor[2] != None:                    
        img[coor[2]:coor[2]+tk,:] = color # up horizontal
    if coor[3] != None:
        img[coor[3]-tk:coor[3],:] = color # down horizontal          

    return img  

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = (height - width)//2
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,\
                                                   pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = (width - height)//2
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,\
                                                   cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square

def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg

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
  
def handwritten_region(hand_img):

    temp_hand = hand_img.copy()
    hand_img_cpy = hand_img.copy()
    temp_hand = cv2.cvtColor(temp_hand, cv2.COLOR_RGB2GRAY) 
    _, temp_hand = cv2.threshold(temp_hand, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  

    img_h, img_w = temp_hand.shape
    temp_hand_ = cv2.erode(temp_hand, np.ones((3,16)), iterations=1)
    x_min, x_max, lines = lines_extraction(temp_hand_, [15, int(img_w/2)])
    for line in lines:
        temp_hand[max(line-10, 0):min(line+10, img_h), 0:img_w] = 255

    temp_ind1 = temp_contour(temp_hand, 2, img_h, img_w, erod_size=0)
    erod_size = 3
    temp_hand = cv2.erode(temp_hand, np.ones((erod_size,erod_size)), iterations=1)
    tk = 2 + erod_size    
    temp_ind2 = temp_contour(temp_hand, tk, img_h, img_w, erod_size=erod_size)
    if len(temp_ind1) > len(temp_ind2): temp_ind = temp_ind1
    else: temp_ind = temp_ind2
        
    if len(temp_ind) > 0:
        if temp_ind[0][3] < 20: temp_ind = temp_ind[1:]

        i = 1
        while 1:
            try:
                x0, w0, y0, h0 = temp_ind[i][0], temp_ind[i][2], temp_ind[i][1], temp_ind[i][3]
                # if w0/h0 > 1.5:
                # if i > 0:
                    # x1, y1 = temp_ind[i-1][0], temp_ind[i-1][1]
                x1, w1, y1, h1 = temp_ind[i-1][0], temp_ind[i-1][2], temp_ind[i-1][1], temp_ind[i-1][3]
                if (x0 - (x1+w1) < 6 or ((x0 - (x1+w1) < 22+erod_size) and w0/h0>1.8)) and min(h0, h1)/max(h0,h1) < 0.7: 
                    temp_ind[i-1] = [min(x0, x1), min(y0, y1), max(x0+w0, x1+w1) - min(x0, x1), max(y0+h0, y1+h1) - min(y0, y1)]
                    del temp_ind[i]    
                    i = i - 1                    
                i = i + 1
            except:
                break
        temp = [hand_img[y:y+h, x:x+w] for [x,y,w,h] in temp_ind]
        return temp
        # temp.append(hand_img[y+5:y+h-5, x+5:x+w-5])
    else: return None
    
def mse(imgA, imgB):

    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def compare_img(img, sample_img_path=None):

    if sample_img_path is None:
        template = cv2.imread(resource_path('config/qr_sample_image.jpg'), 0)
    else:
        template = cv2.imread(sample_img_path, 0) 

    img_h, img_w = img.shape
    tem_h, tem_w = template.shape
    img = cv2.resize(img, None, fx=(tem_w/img_w), fy=(tem_h/img_h))  
               
    m = mse(img, template)
    s = ssim(img, template)

    return s, m
# @profile
def main_parse(filename, template_path, temp_imgs):
    '''
    main parse code
    '''
    image = cv2.imread(filename)
    img_h, img_w, _ = image.shape

    barcode_data, qrcode_data, barcodeOri = file(image)

    qrcode_data, image = barcode_ang(qrcode_data, barcodeOri, img_w, img_h, image)
    barcode_data, _ = barcode_ang(barcode_data, barcodeOri, img_w, img_h, False)
    ##############################################
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
        json_file = template_path/"template.json"
        with open(json_file, "r") as f:
            xyxy = json.load(f)
        total_img = image[xyxy["y0"]:xyxy["y1"], xyxy["x0"]:xyxy["x1"]].copy()
    nm = filename.split('/')[-1]
    flag = cv2.imwrite(f"{temp_imgs}/{nm}", total_img)
    if not flag: cv2.imwrite(f"{temp_imgs}/{nm}", np.ones([100,100,3],dtype=np.uint8)*255)

    del (image, total_img)
    # return 'QR1', 'QR2', 'Bar1', 'Bar2', 'total_value', 'min_prob', 'consider_img'
    if QR1 != 0: return QR1[1]
    elif QR2 != 0: return QR2[1]
    elif Bar1 != 0: return Bar1[1]
    elif Bar2 != 0: return Bar2[1]
    else: return 'Not Recognized'
    
    # QR1 = 'Not Recognized' if QR1 == 0 else QR1[1]
    # QR2 = 'Not Recognized' if QR2 == 0 else QR2[1]
    # Bar1 = 'Not Recognized' if Bar1 == 0 else Bar1[1]
    # Bar2 = 'Not Recognized' if Bar2 == 0 else Bar2[1]
    
    # return QR1, QR2, Bar1, Bar2