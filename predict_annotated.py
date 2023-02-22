import os
import torch
import torchvision
from tqdm import tqdm
import pandas as pd
from glob import glob
import numpy as np
import cv2

from utils import get_temp_model, clean_box, getDF
from utils import StreamArgs, getFrames, tempPredictVideo

torch.cuda.empty_cache()

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def predictNumberBirds():

    n_batch = 4
    num_classes = 2

    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_temp_model(num_classes)
    model.load_state_dict(torch.load(os.getcwd() + '/../models/bird_detection/outputs/models/output_model_temp_full_v0.pt')['model']) #if only CPU available 
    model = model.to(device)
    model.eval()
    print('... model loaded.')
    
    
    output_df = pd.DataFrame(columns = ['date', 'heure', 'methode', 'abondance', 'sous-semis', 'J_T', 'bois',
       'arrosage', 'n°cam'])
    
    root_path = os.getcwd() + '/../data/birds/images'
    folders = glob(root_path + '/*')
    list_files = [glob(f + '/*')[0] for f in folders]

    dict_list = {i+1:[] for i in range(8)}
    for f in list_files:
        dict_list[int(f.split('/')[-1][1])].append(f.split('/')[-2][:6])
        
    

    # iterate over the camera number
    for j in range(8):
        
        c_num = j + 1
        
        print('*************************')
        print('Process camera', c_num)
        print('*************************')
        
        
        # iterate over the dates
        for nv, date in enumerate(dict_list[c_num]):
            
            paths = glob(root_path + '/' + date +'*/C'+ str(c_num) + '*.png')
            print('Video ', nv+1, '/', len(dict_list[c_num]))
            
            out_fns = [paths[0].split('/')[-1][:9] + date , paths[930].split('/')[-1][:9] + date]

            print('Load images...')
            
            img_list = [[cv2.imread(p1) for p1 in paths[:930]], [cv2.imread(p2) for p2 in paths[930:]]]
            
            print('... done.')

            # iterate over right or left
            for i in range(2):
                
                ds = tempPredictVideo(img_list[i], out_fns[i], get_transform())
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
                    
                output_df.to_csv('output_dataset_cameras.csv', index = False)

if __name__ == "__main__":

    predictNumberBirds()