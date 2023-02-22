import os
import torch
import torch.utils.data
from pycocotools.coco import COCO
import numpy as np
import cv2
from datetime import datetime
from datetime import timedelta


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        num_objs = len(coco_annotation)
        
        ## Empty annotation case
        if num_objs == 0:
            # Initialize empty tensors for bboxes and labels
            boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
            labels = torch.ones((0,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.as_tensor(0, dtype=torch.float32)

        else:
            boxes = []
            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Size of bbox (Rectangular)
            areas = []
            for i in range(num_objs):
                areas.append(coco_annotation[i]['area'])
            areas = torch.as_tensor(areas, dtype=torch.float32)
                    
        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img/255.)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

class InferImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths_in, transforms = None):
        self.transforms = transforms
        self.ids = list(paths_in)
        self.eval = eval

    def __getitem__(self, index):
        # path for input image
        path = self.paths_in[index]
        # open the input image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(img/255.)
        return img

    def __len__(self):
        return len(self.ids)

def getBefAf(path, root, img):
    
    time = path.split('_')[-1].split('.')[0]
    datetime_object = datetime.strptime(time, '%Hh%M')

    ## add or remoce 1 minute
    t_after = datetime_object + timedelta(minutes=1)
    t_before = datetime_object + timedelta(minutes=-1)

    # Convert datetime object to string in specific format and replace
    aft_path = path.replace(time, t_after.strftime('%Hh%M'))
    bef_path = path.replace(time, t_before.strftime('%Hh%M'))
        
    if time == '06h00':
        img_before = np.zeros(img.shape)
    else:
        img_before = cv2.imread(os.path.join(root, bef_path))
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    if time == '21h29':
        img_after = np.zeros(img.shape)
    else:
        img_after = cv2.imread(os.path.join(root, aft_path))
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    
    return img_before, img_after
    

class tempInferDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ## convert to datetime obj
        img_before, img_after = getBefAf(path, self.root, img)

        num_objs = len(coco_annotation)
        
        ## Empty annotation case
        if num_objs == 0:
            # Initialize empty tensors for bboxes and labels
            boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
            labels = torch.ones((0,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.as_tensor(0, dtype=torch.float32)

        else:
            boxes = []
            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # Size of bbox (Rectangular)
            areas = []
            for i in range(num_objs):
                areas.append(coco_annotation[i]['area'])
            areas = torch.as_tensor(areas, dtype=torch.float32)
           
        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        
        stack_img = np.concatenate([img_before, img, img_after], axis = 2)

        if self.transforms is not None:
            img = self.transforms(stack_img/255.)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)



class tempPredictDataset(torch.utils.data.Dataset):
    def __init__(self, root, paths_in, transforms = None):
        self.root = root
        self.paths_in = paths_in
        self.transforms = transforms
        self.ids = list(paths_in)

    def __getitem__(self, index):
        # path for input image
        path = self.paths_in[index]
        fn = path.split('/')[-1].split('.')[0]
        # open the input image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_before, img_after = getBefAf(path, self.root, img)
        
        stack_img = np.concatenate([img_before, img, img_after], axis = 2)
        
        if self.transforms is not None:
            img = self.transforms(stack_img/255.)
            
        return img, fn

    def __len__(self):
        return len(self.ids)

def getOtherImg(img_list, ind):
    if ind == 0:
        img_bef = np.zeros(img_list[0].shape)
    else:
        img_bef = img_list[ind-1]
    if ind == 929:
        img_af = np.zeros(img_list[0].shape)
    else:
        img_af = img_list[ind + 1]
    return img_bef, img_af

def getTime(ind):
    
    init_time = '06h00'
    datetime_object = datetime.strptime(init_time, '%Hh%M')

    ## add or remoce 1 minute
    new_time = datetime_object + timedelta(minutes = ind)

    # Convert datetime object to string in specific format and replace
    time = new_time.strftime('%Hh%M')
    
    return time

class tempPredictVideo(torch.utils.data.Dataset):
    def __init__(self, img_list, out_fn, transforms = None):
        self.img_list = img_list
        self.filename = out_fn
        self.transforms = transforms

    def __getitem__(self, index):
        # path for input image
        fn = self.filename + '_' + getTime(index)
        # open the input image
        img = self.img_list[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_before, img_after = getOtherImg(self.img_list, index)
        
        stack_img = np.concatenate([img_before, img, img_after], axis = 2)
        
        if self.transforms is not None:
            img = self.transforms(stack_img/255.)
            
        return img, fn

    def __len__(self):
        return len(self.img_list)
