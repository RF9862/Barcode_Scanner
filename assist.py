
import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import time
from keras import backend as K

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import cv2
from keras.models import load_model
import numpy as np

@smart_inference_mode()
def getTotalValue(
        weights='best.pt',  # model path or triton URL
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='outputs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        main_self = None
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    result = []
    nn = 0
    for path, im, im0s, vid_cap, s in dataset:
        nn += 1
        if nn % 200 == 0: K.clear_session()
        box, con, labels = [], [], []
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=False) if visualize else False
            pred = model(im, augment=augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    box.append(xyxy)
                    con.append(float(conf))
                    labels.append(str(int(cls)))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n') 

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # label = names[c]
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        ## applying model.h5
        digit_cnt = len(box)
        try:
            xc = [int((bo[0]+bo[2])/2) for bo in box]
            xc, box, con, labels = list(zip(*sorted(zip(xc, box, con, labels))))
            ## finding differences between neibour values in xc ##
            xc_diff = [j-i for i, j in zip(xc[:-1], xc[1:])]
            gap_lim = 20
            del_list = []
            for i, xcd in enumerate(xc_diff):
                if xcd < gap_lim:
                    if con[i] > con[i+1]: del_list.append(i+1)
                    else: del_list.append(i)
            labels = [v for i, v in enumerate(labels) if i not in del_list]
            con = [v for i, v in enumerate(con) if i not in del_list]
            xc = [v for i, v in enumerate(xc) if i not in del_list]
            new_box = [v for i, v in enumerate(box) if i not in del_list]
            ## check according to yc
            if len(xc) > 2:
                yc = [int(v[1]/2)+int(v[3]/2) for v in new_box]
                ycMedian = np.median(yc)
                if abs(yc[0]-ycMedian) > 50:
                    labels, con, xc, new_box = labels[1:], con[1:], xc[1:], new_box[1:]
                # ycIndForRemove = [i for i,v in enumerate(yc) if abs(v-ycMedian) > 40]
            ''' ## post processing ##'''
            # 1) rechecking value of 0 in labels 
        except: pass
        try:
            for i, lab in enumerate(labels[0:4]):
                # if (lab == "0" and i<2 or lab == '7') and con[i] < 0.92 or (lab == "5" or lab == "8") and con[i] < 0.85: 
                if con[i] < 0.9 or i==2 and lab != '0':
                    pred, prob = Recognize_Digit(im0s[int(new_box[i][1]):int(new_box[i][3]), int(new_box[i][0]):int(new_box[i][2])])  
                    labels[i] = str(pred)
                    con[i] = prob             
            if  len(labels) < 4:
                ## finding ref yc ##
                yc = [int((bo[1]+bo[3])/2) for bo in box]
                yc.sort()
                yc,  ycnt= subset(yc, 15, 'medi')
                im0sTemp = im0s.copy()
                try:
                    refYc = yc[ycnt.index(max(ycnt))]         
                    ycLim, hLim, tk = 15, 20, 3
                    for bo in box:
                        im0sTemp[int(bo[1])-tk:int(bo[3])+tk, int(bo[0])-tk:int(bo[2])+tk] = [255,255,255]
                except: ycLim, hLim, tk, refYc = 1000, 20, 3, 0
                gray = cv2.cvtColor(im0sTemp, cv2.COLOR_RGB2GRAY)
                _, bin = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                cnt, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                imgcpy = im0s.copy()
                for c in cnt:
                    x, y, w, h = cv2.boundingRect(c)
                    if h > hLim and w < 150 and abs(refYc-(y+h/2)) < ycLim:# and w < 60:# and h >15:
                        # cv2.rectangle(xx, (x, y), (x + w, y + h), (0, 255, 0),1)   
                        W = [int(w/2), w] if w>90 else [w]
                        for w in W:
                            pred, prob = Recognize_Digit(im0s[y:y+h, x:x+w])  
                            if prob > 0.4:
                                con.append(prob)
                                labels.append(str(pred))
                                xc.append(int(x+w/2))
                                new_box.append([x, y, x+w, y+h])
                                image = cv2.rectangle(imgcpy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            x = x+w
                xc, labels, new_box, con = list(zip(*sorted(zip(xc, labels, new_box, con))))                
            labels = list(labels)
            con = list(con)
            if len(labels) < 4: 
                labels = labels + ['0', '0', '0']
                con = con + [1,1,1]
            if labels[2] != '5': # third number is alwasy 0 or 5.
                labels[2] = '0'
            con[3], con[2] = 1, 1 # last number is always zero.  
            labels[3] = '0'      
            labels.insert(2, '.')
            print(f"===== {labels[0:5]} ====")
            result.append([''.join(labels[0:5]),min(con)])
            # Print time (inference-only)
            s1 = s.split(' ')
            cnt = int(s1[1].split('/')[0])
            total = int(s1[1].split('/')[1])
            if main_self:
                main_self.cnt2 = int(20 + main_self.total2*0.8/len(dataset) *nn)
                # main_self.total2=total
                time.sleep(0.2)
                main_self.single_done2.emit()
                
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        except: result.append(['-', 1])
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return result

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
# @profile
def Recognize_Digit(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # img_h, img_w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        roi = blurred#[yy:yy + hh, xx:xx + ww]
        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        roi = makeSquare(roi)
        roi = resize_to_pixel(28, roi)
        roi = roi / 255.0       
        roi = roi.reshape(1,28,28,1)             
        
        ## Get Prediction
        predict_x = digit_model.predict(roi, 1, verbose = 0)[0]
        final_pred=np.argmax(predict_x)
        prob = max(predict_x)
    except:
        final_pred = None    
    # del model
    return final_pred, prob
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

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

digit_model = load_model(resource_path('weights')+'/model.h5')    

