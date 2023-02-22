import os
import torch
import torch.utils.data
from pycocotools.coco import COCO
import numpy as np
import cv2
import albumentations as A
from datetime import datetime
from datetime import timedelta

class myOwnDataset(torch.utils.data.Dataset):
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
        
        augmentation = A.Compose([
                A.RandomSizedBBoxSafeCrop(width=450, height=450, erosion_rate=0.2, p = 0.7),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(format='coco', min_visibility = 1)
            )
        
        ## Empty annotation case
        if len(coco_annotation) == 0:
            # Initialize empty tensors for bboxes and labels
            boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
            labels = torch.ones((0,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.as_tensor(0, dtype=torch.float32)

        else:
            ## do augmentation
            transformed = augmentation(image=img, bboxes=[c['bbox']+['bird']for c in coco_annotation])
            anno_aug = [t[:4] for t in transformed['bboxes']]
            img = transformed['image']
            
            # number of objects in the image
            num_objs = len(anno_aug)
            
            if num_objs == 0:
                # Initialize empty tensors for bboxes and labels
                boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
                labels = torch.ones((0,), dtype=torch.int64)
                # Iscrowd
                iscrowd = torch.zeros((0,), dtype=torch.int64)
                areas = torch.as_tensor(0, dtype=torch.float32)
            
            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            else:
                boxes = []
                for i in range(num_objs):
                    xmin = anno_aug[i][0]
                    ymin = anno_aug[i][1]
                    xmax = xmin + anno_aug[i][2]
                    ymax = ymin + anno_aug[i][3]
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

class tempDataset(torch.utils.data.Dataset):
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
            img_before = cv2.imread(os.path.join(self.root, bef_path))
            img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        if time == '21h29':
            img_after = np.zeros(img.shape)
        else:
            img_after = cv2.imread(os.path.join(self.root, aft_path))
            img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        augmentation = A.Compose([
                A.RandomSizedBBoxSafeCrop(width=450, height=450, erosion_rate=0.2, p = 0.7),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(format='coco', min_visibility = 1),
            additional_targets={'image0': 'image', 'image1': 'image'}
            )
        
        ## Empty annotation case
        if len(coco_annotation) == 0:
            # Initialize empty tensors for bboxes and labels
            boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
            labels = torch.ones((0,), dtype=torch.int64)
            # Iscrowd
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.as_tensor(0, dtype=torch.float32)

        else:
            ## do augmentation
            transformed = augmentation(image0 = img_before, image=img, image1 = img_after, 
                                       bboxes=[c['bbox']+['bird']for c in coco_annotation])
            anno_aug = [t[:4] for t in transformed['bboxes']]
            img_before, img, img_after = transformed['image0'],transformed['image'], transformed['image1']
            
            # number of objects in the image
            num_objs = len(anno_aug)
            
            if num_objs == 0:
                # Initialize empty tensors for bboxes and labels
                boxes = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
                labels = torch.ones((0,), dtype=torch.int64)
                # Iscrowd
                iscrowd = torch.zeros((0,), dtype=torch.int64)
                areas = torch.as_tensor(0, dtype=torch.float32)
            
            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            else:
                boxes = []
                for i in range(num_objs):
                    xmin = anno_aug[i][0]
                    ymin = anno_aug[i][1]
                    xmax = xmin + anno_aug[i][2]
                    ymax = ymin + anno_aug[i][3]
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
            img = self.transforms(stack_img / 255.)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)