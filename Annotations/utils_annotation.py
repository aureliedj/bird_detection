import cv2 
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

def getListImg(path):
    
    """Opens an AVI file and convert to list of images.
    
    Inputs:
        path (str): file path

    Returns:
        frames (list of np arrays): list of the 1020 images in the timelapse
        count (int): number of images retrieved in the videofile
    """
    
    cap = cv2.VideoCapture(path)
    
    frames = []
    success = 1
    count = 0
    speed = 1

    while success:
        success, image = cap.read()
        if(count % speed == 0):
            frames.append(image)
        count += 1
    return frames, count

def saveFrames(inpath, outpath, filename, Y, m, d, save = True):
    """Save the frames of a timelapse video and return the filenames.

    Args:
        inpath (str): path to timelapse file .AVI
        outpath (str): path to the output directory
        filename (str): filename 'C#_TC00##' where # is the camera number, ## the video number
        Y (int): year of acquisition
        m (int): month of acquisition
        d (int): day of acquisition
        save (bool, optional): If True, save the frames in the pathout directory. Defaults to True.

    Returns:
        out (list): list of filenames
    """
    #Takes around 11s for 1 file
    frames, count = getListImg(inpath)

    H, M = 4, 0
    out = []
    pbar = tqdm(frames[:1020])

    for f in pbar:
        if M % 60 == 0:
            H += 1
            M = 0
        date = dt.datetime(Y,m,d,H,M)
        suff = date.strftime('%y-%m-%d_%Hh%M')
        out.append(filename+'_'+suff+'.png')
        
        pbar.set_description(f"Saving frame {date.strftime('%Y-%m-%d %H:%M')}    ")
        if save:
            cv2.imwrite(outpath+'\\'+filename+'_'+suff+'.png', f )
        M += 1
    return out

def getImgAnno(paths, coco_img, ID_init):
    """Compute annotations to save in the coco_data['images] from a list of the saved frames' paths

    Args:
        paths (list): list of paths to images in one frame
        coco_img (list): list of dict of the coco_data['images'] annotations info
        ID_init (int): 7 digits number CVVSSSS 
                       where C:camera [1-4], VV:video [00-38], SSSS:slice [1-1020]
    Returns:
        coco_img(list): list of dict of the coco_data['images'] annotations info
    """
    for i, p in enumerate(paths):
    
        ID = ID_init + i
        fn = p.split('\\')[-1]
        D = fn.split('_')[2]
        H = fn.split('_')[-1][:2]
        M = fn.split('_')[-1][3:5]
        
        coco_img.append({
            "file_name":fn,
            'file_path':p,
            'height':720,
            'width':1280,
            'date_capture':D + ' ' + H+':'+M,
            'id':ID
        })
        
    return coco_img

def getAnno(pathin, dict_img):
    """Create annotations from markers files and coco_data['images'] dictionnary

    Args:
        pathin (str): markers .csv file path
        dict_img (dict): dictionnary of coco dataset image attributes

    Returns:
        df_anno (df): dataframe of annotations
    """
    
    dict_id2fn = {c['id']:c['file_name'] for c in dict_img}
    
    #read markers csv file
    df = pd.read_csv(pathin)
    #get frame slice index, X, Y 
    slice = df['Slice'].values
    X, Y = df['X'].values.reshape(-1,1), df['Y'].values.reshape(-1,1)
    #Compute fields
    filenames = np.array([dict_id2fn[2200000+s] for s in slice]).reshape(-1,1)
    width = np.ones((len(slice),1)).astype('uint')*720
    height = np.ones((len(slice),1)).astype('uint')*1280
    #linear regression y = 0,0648x - 3,662
    delta = 0.0648*Y- 3.662
    delta = delta.astype('uint')
    xmin, xmax = X-delta , X+delta
    ymin, ymax = Y-delta, Y+delta
    #create classes
    classes = []
    for i in range(len(slice)):
        classes.append('bird')
    classes = np.array(classes).reshape(-1,1)
    #create dataframe
    arr = np.concatenate([filenames,width,height,classes,xmin,ymin,xmax,ymax],axis=1)
    cols=['filename','width','height','class','xmin','ymin','xmax','ymax']
    df_anno = pd.DataFrame(arr, columns = cols)
    
    return df_anno
