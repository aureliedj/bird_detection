import numpy as np
import cv2
import pandas as pd
import os
from glob import glob


def cropImg(img, points):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255))
    
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

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

def getDate(c_num):
    list_dates = ['220429', '220430', '220501', '220502', '220503', '220504', '220505', '220506', '220507',
                  '220508', '220509', '220510', '220511', '220512', '220513']
    
    file_date = [f for f in list_dates]
    
    return file_date

class StreamArgs():
    def __init__(self, c_num, path, n):
        self.c_num = c_num 
        
        root = os.getcwd()
        #change path to the folder where you have timelapse video
        pathin = root + '/../data/timelapse/cam_'+ str(self.c_num) 
        
        path_coord_left = pathin +'/../xy_coordinates_left.csv' #path to coordinates
        path_coord_right = pathin + '/../xy_coordinates_right.csv' #path to coordinates
        
        self.dict_suf = {1:['C1_AA_ST_','C1_AP_SN_'], #file names [left, right]
                    2:['C2_AP_SN_','C2_AA_ST_'],
                    3:['C3_AP_SD_','C3_AP_BF_'],
                    4:['C4_AP_BF_','C4_AP_SD_'],
                    5:['C5_AA_SD_','C5_AP_SN_'],
                    6:['C6_AP_SN_','C6_AA_SD_'],
                    7:['C7_AA_BF_','C7_AP_ST_'],
                    8:['C8_AP_ST_','C8_AA_BF_']}
        
        self.path = path
        self.date = getDate(c_num)[n]
        
        self.out_fns = [self.dict_suf[self.c_num][i] + self.date for i in range(2)]
            
        self.dict_pts_left = read_coord(path_coord_left)

        #add manually points where crop more
        self.dict_pts_left[3] = np.array([[247,702],[150,530],[134,368],[170,220],
                              [286,68],[376,5],[573,12],[563,702]])
        self.dict_pts_left[6] = np.array([[315,702],[200,559],[159,370],[212,168],
                              [366,92],[612,83],[608,702]])
        self.dict_pts_left[8] = np.array([[304,702],[206,569],[162,386],[213,138],
                              [329,90],[579,100],[579,702]])
        self.dict_pts_right = read_coord(path_coord_right)
        self.dict_pts_right[3] = np.array([[563,702],[573,12],[808,28],[977,136],
                              [1050,270],[1070,419],[1041,582],[965,702]])
        self.dict_pts_right[6] = np.array([[608,702],[612,83],[844,89],[1081,226],
                              [1091,339],[1079,463],[1003,636],[934,702]])
        self.dict_pts_right[8] = np.array([[579,702],[579,100],[788,105],[1126,314],
                              [1122,417],[1097,529],[1038,646],[973,702]])
        
def read_coord(path):
    #Read coordinates file for cropping
    coordinates = pd.read_csv(path, sep = ';')
    cols = ['camera'] + list(coordinates.columns)[1:]
    coordinates.columns = cols    
                
    dict_pts = {}

    for i, row in coordinates.iterrows():
        dict_pts[row[cols[0]]]=np.array([ [row[cols[1]],row[cols[2]]],  [row[cols[3]],row[cols[4]]],  
                                    [row[cols[5]],row[cols[6]]], [row[cols[7]],row[cols[8]]],  [row[cols[9]],row[cols[10]]] ])
    return dict_pts

def getFrames(self):
    
    #Read .AVI files
    print('Read frames ...')
    frames, count = getListImg(self.path)
    
    output_frames = []
    
    #Select images from 06:00 a.m.(60 below) to 09:30 p.m. (990 below)
    for i in np.arange(2):
        if i == 0:
            print('Crop left side ...')
            crop_frames = [cropImg(img, self.dict_pts_left[self.c_num]) for img in frames[60:990]]
        else:
            print('Crop right side...')
            crop_frames = [cropImg(img, self.dict_pts_right[self.c_num]) for img in frames[60:990]]
        output_frames.append(crop_frames)
        
    return output_frames