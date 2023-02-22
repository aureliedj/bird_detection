import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import get_model_object_detection, myOwnDataset, InferDataset, get_iou, compute_metrics, get_all_metrics


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    
    root = os.getcwd() + '/../data/birds/annotations/all'
    pathin = root + '/../full_coco_train.json'
    pathin_val = root + '/../full_coco_val.json'
    n_batch = 4


    ds = myOwnDataset(root, pathin, get_transform())
    ds_val = InferDataset(root, pathin_val, get_transform())

    dataloader = torch.utils.data.DataLoader(ds,
                batch_size = n_batch,
                shuffle = True,
                num_workers = 6,
                collate_fn = collate_fn)
    
    dataloader_val = torch.utils.data.DataLoader(ds_val,
                batch_size = n_batch,
                shuffle = False,
                num_workers = 6,
                collate_fn = collate_fn)
    
    # 2 classes; Only target class or background
    num_classes = 2
    num_epochs = 20
    

    # 2 classes; Only target class or background
    model = get_model_object_detection(num_classes)

    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # move model to the right device
    model.to(device)   
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)

    n_batch_per_epoch = int(len(ds) / n_batch)
    n_batch_per_epoch_val = int(len(ds_val)/n_batch)
    
    save_metrics = []
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        avg_loss = 0.0
        val_loss = 0.0
        train_metrics = []
        val_metrics = []
        dump_period = 2
        
        print('EPOCH ', epoch)
        progress_bar = tqdm(enumerate(dataloader), total=n_batch_per_epoch) 
        for batch_id, data in progress_bar:
            model.train()
                
            imgs, annotations = data
            imgs = list(img.float().to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss_dict.values())
            
            avg_loss += losses.to('cpu').detach().numpy()
            train_loss = avg_loss / (batch_id + 1)

            if batch_id % dump_period == 0:
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            
            if batch_id % 4 == 0: 
                train_metrics.append(compute_metrics(model, imgs, annotations))
            
            # print running loss
            progress_bar.set_postfix(loss = train_loss)
            # print running loss
            progress_bar.set_postfix(loss = train_loss, 
                                    prec = np.mean(train_metrics, axis = 0)[1], 
                                    acc = np.mean(train_metrics, axis = 0)[-1])
            
        
        progress_val = tqdm(enumerate(dataloader_val), total=n_batch_per_epoch_val)  
        for batch_id, data in progress_val:
            model.train()
                
            imgs, annotations = data
            imgs = list(img.float().to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            val_loss_dict = model(imgs, annotations)
            val_losses = sum(val_loss_dict.values())
            val_loss += val_losses.to('cpu').detach().numpy()
            val_mean_loss = val_loss / (batch_id + 1)
            
            val_metrics.append(compute_metrics(model, imgs, annotations))
            
            # print running loss
            progress_val.set_postfix(loss = val_mean_loss, 
                                     prec = np.mean(val_metrics, axis = 0)[1], 
                                     acc = np.mean(val_metrics, axis = 0)[-1])
            torch.cuda.empty_cache()
        
        val_metrics = np.stack(val_metrics)
        count_v = np.count_nonzero(val_metrics[:,0])
        if count_v != 0:
            val_iou = np.sum(val_metrics[:,0])/count_v
        else:
            val_iou = 0.
        val_prec, val_rec, val_acc = np.mean(val_metrics[:,1:], axis = 0)     
        
        train_metrics = np.stack(train_metrics)
        count_t = np.count_nonzero(train_metrics[:,0])
        if count_t != 0:
            train_iou =  np.sum(train_metrics[:,0])/count_t
        else:
            train_iou = 0.
        train_prec, train_rec, train_acc = np.mean(train_metrics[:,1:], axis = 0)  
        
        save_metrics.append([train_loss, val_mean_loss, val_iou, train_iou, val_prec, train_prec, val_rec, train_rec, val_acc, train_acc])
        
        df_loss = pd.DataFrame(np.array(save_metrics).reshape(-1,10), 
                            columns = ['train loss','val loss','val iou', 'train iou', 
                                        'val prec', 'train prec', 'val rec','train rec', 'val acc', 'train acc'])
        df_loss.to_csv('output_metrics_full_v2.csv', index = False)
            
        if epoch == 0:
            out_loss = val_mean_loss
            
        if val_mean_loss <= out_loss:
            # save the last checkpoint in a separate file
            last_checkpoint = {'model': model.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'epoch': epoch,
                                            'model_params': {}} #TEMPORARY
            torch.save(last_checkpoint, os.getcwd() + '/output_model_full_v2.pt')
            out_loss = val_mean_loss
        torch.cuda.empty_cache()

if __name__ == "__main__":

    train()