import os
import torch
import torchvision
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from glob import glob
from datetime import datetime

from utils import get_temp_model, tempPredictDataset, IoU, clean_box, get_birds, plot_box, getDF
from utils import getListImg, cropImg, StreamArgs, getFrames, tempPredictVideo

torch.cuda.empty_cache()

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def predictNumberBirds():

    # parameters TO CHANGE
    num_camera = 8
    path_to_trained_model = os.getcwd() + '/../models/bird_detection/outputs/models/output_model_temp_full_v0.pt'
    path_to_timelapse_folders = os.getcwd() + '/../data/timelapse/cam_'
    output_filename = 'results_all_cameras.csv'

    # other parameters
    n_batch = 4
    num_classes = 2

    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_temp_model(num_classes)
    model.load_state_dict(torch.load(path_to_trained_model)['model'])
    model = model.to(device)
    model.eval()
    print('... model loaded.')
    
    output_df = pd.DataFrame(columns = ['date', 'heure', 'methode', 'abondance', 'sous-semis', 'J_T', 'bois',
       'arrosage', 'n°cam'])

    # iterate over the camera number
    for j in range(num_camera):
        
        c_num = j + 1
        
        print('*************************')
        print('Process camera', c_num)
        print('*************************')
        
        pathin = path_to_timelapse_folders + str(c_num) 
        paths = glob(pathin + '/*.AVI')
        num_video = len(paths)
        
        # iterate over the dates
        for nv, p in enumerate(paths):
            
            print('Video ', nv + 1, '/', num_video)
            
            args = StreamArgs(c_num, p, nv)
            img_list = getFrames(args)

            # iterate over right or left
            for i in range(2):
                
                ds = tempPredictVideo(img_list[i], args.out_fns[i], get_transform())
                dataloader = torch.utils.data.DataLoader(ds,
                            batch_size = n_batch,
                            shuffle = False,
                            num_workers = 6,
                            collate_fn = collate_fn)
                # iterate over video frames
                for data in tqdm(dataloader):
            
                    imgs, fns = data
                    imgs = list(img.float().to(device) for img in imgs)
                    fns = list(fns)
                    
                    pred = model(imgs)
                    
                    # keep predictions of scores higher than 0.6
                    boxes = []
                    for j, p in enumerate(pred):
                        cond = p['scores'] > 0.6
                        boxes.append(p['boxes'][cond])
                        
                    # remove boxes prediction that overlap more than 75%
                    clean_boxes = clean_box(boxes)
                    
                    # save all information in dataframe
                    df = getDF(clean_boxes, fns)
                    output_df = pd.concat([output_df, df])
                    
                output_df.to_csv(output_filename, index = False)

if __name__ == "__main__":

    predictNumberBirds()