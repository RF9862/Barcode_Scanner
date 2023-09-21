from barcode_post import post_processing
import datetime
import os, gc, sys
# import logging
from barcode_par import main_parse
import shutil
from pathlib import Path
from assist import getTotalValue
import time
# logger=None
def clear_contents(dir_path):
    '''
    Deletes the contents of the given filepath. Useful for testing runs.
    '''
    filelist = os.listdir(dir_path)
    if filelist:
        for f in filelist:
            if os.path.isdir(os.path.join(dir_path, f)):
                shutil.rmtree(os.path.join(dir_path, f))
            else:
                os.remove(os.path.join(dir_path, f))
    return None

def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def main(main_self, data_dirs, output_dir, err_dir, cropPathBody, save_path):
    '''
    Main control flow:
        1. Checks if required folders exist; if not, creates them
        2. Loops over each PDF file in data_path and calls parse_doc().
        3. Output xlsx files are written to output_path.
    '''
    gc.collect()
    # Check if organizing folders exist
    makedir(err_dir)
    clear_contents(err_dir)
    temp_imgs = os.path.join(output_dir, cropPathBody)
    print(temp_imgs)
    makedir(temp_imgs)

    # clear_contents(temp_imgs)
    weights = 'weights/epoch40.pt'
    # Get list of pdfs to parse
    # img_list = [f for f in os.listdir(data_dir) if (f.split('.')[-1].lower() in ['png','jpg', 'tif'])]
    # img_list.sort()
    # logger.info(f"{len(img_list)} file(s) detected.")

    vals = []
    fail_cnt = 0
    start = datetime.datetime.now()
    template_path = 'config'
    img_list = []
    success_lists = []
    for data_dir in data_dirs:
        img_list = img_list + [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.split('.')[-1].lower() in ['png','jpg', 'tif'])]    
    main_self.total2 = 100
    main_self.cnt2 = 0
    for cnt, im in enumerate(img_list):
        im = im.replace('\\', '/')
        
        try:
            imBody = im.split('/')[-1]
            vals.append([main_parse(im, template_path, temp_imgs), imBody])
            success_lists.append(output_dir.replace('\\', '/') + "/input/" + im.split('/')[-1])
            main_self.cnt2 = int(main_self.total2/5/len(img_list) * (cnt + 1)) 
            main_self.single_done2.emit()
        except Exception as e:
            fail_cnt += 1
            shutil.copyfile(im, os.path.join(err_dir, imBody))
            print(e)
        
        gc.collect()

    labels = getTotalValue(weights=weights, source=temp_imgs, conf_thres=0.3, project=Path(output_dir)/"detect", main_self=main_self)
    post_processing(vals, labels, success_lists, temp_imgs, save_path)
    print(f"Success: {len(img_list)-fail_cnt}, Failed: {fail_cnt}")
    # logger.info(f"Success: {len(img_list)-fail_cnt}, Failed: {fail_cnt}")
    duration = datetime.datetime.now() - start
    # logger.info(f"Time taken: {duration}")    
    print(f"Time taken: {duration}")
    # shutil.rmtree(temp_imgs)
    # main_self.openFolder2(save_path)
    gc.collect()        

    return vals, labels, success_lists

#if __name__ == "__main__":
def barcode(main_self, img_list, out, cropPathBody, save_path):
    # Key paths and parameters
    # DATA_DIR = folder #"bar_inputs"
    OUTPUT_DIR = out
    ERR_DIR = OUTPUT_DIR+"/failed"
    # folderBody = folder.split('\\')[-1]
    # global logger
    # loggerPath = os.path.join(out, f'user.log')
    # Initialize logger
    # if os.path.exists(loggerPath):
    #     os.remove(loggerPath)
    # logger = logging.getLogger('user')
    # logger.setLevel(logging.INFO)
    # ch = logging.StreamHandler()
    # fh = logging.FileHandler(loggerPath)
    # logger.addHandler(ch)
    # logger.addHandler(fh)
    
    # Run main control flow    
    vals, labels, imgs = main(main_self, img_list, OUTPUT_DIR, ERR_DIR, cropPathBody, save_path)

    return vals, labels, imgs
