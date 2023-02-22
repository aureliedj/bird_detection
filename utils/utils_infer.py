import numpy as np
from utils import IoU
import pandas as pd

def clean_box(predictions):
    out_boxes = []

    # Iterate over the predicted boxes
    for prediction in predictions:
        # Initialize a flag to check if any predicted box matches a ground truth box
        save = []
        remove = []
        
        for i, box in enumerate(prediction):
            # Iterate over the ground truth boxes
            for j, pred in enumerate(prediction):
                    # Compute IoU between the predicted box and the ground truth
                iou = IoU(box, pred)
                
                if iou > 0.2:
                    if i not in remove:
                        save.append(i)
                    if j not in save:
                        remove.append(j)
                        
        out_boxes.append([prediction[i] for i in np.unique(save)])
        
    return out_boxes

def get_birds(boxes):
    return [len(b) for b in boxes]

def getDF(clean_boxes, fns):
    
    date = [f.split('_')[-2] for f in fns]
    time = [f.split('_')[-1] for f in fns]
    cam = [int(f.split('_')[0][1]) for f in fns]
    meth = [f.split('_')[2] for f in fns]
    ssemis = [f.split('_')[1] for f in fns]
    counts = get_birds(clean_boxes)

    dict_bois = {1:'oui',
                2:'non',
                3:'non',
                4:'oui',
                5:'oui',
                6:'non',
                7:'oui',
                8:'non'}

    dict_arro = {1:'non',
                2:'oui',
                3:'oui',
                4:'non',
                5:'non',
                6:'oui',
                7:'non',
                8:'oui'}

    date_semis = '220428'
    
    arr = np.stack([date, time, meth, counts, ssemis, date, cam, cam, cam], axis = 1)
    df = pd.DataFrame(arr, columns = ['date', 'heure', 'methode', 'abondance', 'sous-semis','J_T', 'bois', 'arrosage', 'n°cam'])

    df['arrosage'] = df['arrosage'].astype('int')
    df['arrosage'] = df['arrosage'].map(dict_arro)
    df['n°cam'] = df['n°cam'].astype('int')
    df['bois'] = df['bois'].astype('int')
    df['bois'] = df['bois'].map(dict_bois)


    df['date'] = pd.to_datetime(df['date'], format = '%y%m%d')
    df['J_T'] = pd.to_datetime(['220428' for i in range(len(df))], format = '%y%m%d')
    df['J_T'] = (df['date'] - df['J_T']).dt.days
    df['heure'] = pd.to_datetime(df['heure'], format= '%Hh%M').dt.time
    df.loc[df['methode']=='SN', 'sous-semis'] = 'SN'
    
    return df