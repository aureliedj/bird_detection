
import numpy as np

def IoU(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_iou(prediction, gt_boxes):
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    if len(gt_boxes) > 0:
        # Initialize a flag to check if any predicted box matches a ground truth box
        results = np.zeros((len(prediction), len(gt_boxes)))

        # Iterate over the predicted boxes
        for i, box in enumerate(prediction):
            # Iterate over the ground truth boxes
            for j, gt_box in enumerate(gt_boxes):
                # Compute IoU between the predicted box and the ground truth
                results[i,j] = IoU(box, gt_box)
        
        best_iou = np.zeros((len(gt_boxes),1))
        for i, n in enumerate(np.argmax(results, axis = 1)):
            if results[i, n] > best_iou[n]:
                best_iou[n] = results[i, n]
        
        if np.all(best_iou == 0):
            avg_iou = 0.0
            
        else:            
            avg_iou = np.mean(best_iou[best_iou!=0])
        
        tp = np.count_nonzero(best_iou)
        fn = len(gt_boxes) - tp
        fp = len(prediction) - np.count_nonzero(np.max(results, axis = 1))
        tn = 0
        
        if tp == 0:
            prec = 0
            rec = 0
            acc = 0
        else:  
            prec = tp / (tp + tn)
            rec = tp / (tp + fp)
            acc = (tp+tn)/(tp+tn+fp+fn)
    
    else:
        avg_iou, best_iou, conf_mat = 0, 0, 0
        if len(prediction) == 0:
            prec = 1
            rec = 1
            acc = 1
        else:
            prec = 0
            rec = 0
            acc = 0
        
    return avg_iou, prec, rec, acc

def compute_metrics(model, imgs, annotations):
    
    box_iou, count = 0., 0
    avg_prec, avg_rec, avg_acc = 0., 0., 0.
    
    model.eval()
    pred = model(imgs)
    N = len(pred)
    
    for j, p in enumerate(pred):
        cond = p['scores'] > 0.6
        boxes = p['boxes'][cond]
        gt_box = annotations[j]['boxes']

        mean_iou, prec, rec, acc = get_iou(boxes, gt_box)
        
        if mean_iou != 0:
            count += 1
            
        box_iou += mean_iou
        avg_prec += prec
        avg_rec += rec
        avg_acc += acc
    
    if count != 0:
        miou = box_iou / count
    else:
        miou = 0.
        
    return miou, avg_prec/N, avg_rec/N, avg_acc/N

def get_all_metrics(box_iou, count, cm):
    
    if count != 0:
        val_iou = box_iou / count
    else:
        val_iou = np.NaN
    val_prec = cm[0,0] / np.sum(cm[0,:])
    val_rec = cm[0,0] / np.sum(cm[:,0])
    val_acc = sum(np.diag(cm))/np.sum(cm)
    
    return val_iou, val_prec, val_rec, val_acc