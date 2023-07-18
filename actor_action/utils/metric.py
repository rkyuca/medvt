import numpy as np
import math
import torch
import torch.nn.functional as F

def db_eval_iou(annotation, segmentation):
    """
    Collected from https://github.com/fperazzi/davis/blob/main/python/lib/davis/measures/jaccard.py
    Compute region similarity as the Jaccard Index.
         Arguments:
             annotation   (ndarray): binary annotation   map.
             segmentation (ndarray): binary segmentation map.
         Return:
             jaccard (float): region similarity
    """
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
               np.sum((annotation | segmentation), dtype=np.float32)


def db_eval_multiclass_iou(label, pred, num_classes=19):
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)

    #TODO: only evaluate on the valid classes+foreground from the A2D readme file.
    valid_classes = [11.0,12.0,13.0,15.0,16.0,17.0,18.0,19.0,21.0,22.0,26.0,28.0,29.0,\
        34,35,36,39,41,43,44,45,46,48,49,54,55,56,57,\
        59,61,63,65,66,68,69,72,73,75,76,77,78,79]

    # for sem_class in range(num_classes):
    for sem_class in range(1, num_classes): # skip the background index, which is 0
    # for sem_class in valid_classes:
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)

        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    # breakpoint()
    return np.mean(present_iou_list)

def new_iou(label, pred, num_classes=80):
    
    # skip the first index '0', which is the background index
    label = label[:, 1:]
    pred = pred[:, 1:]

    # inter and union
    inter = (label * pred).sum(-1)
    union = torch.any(torch.stack(
        [label, pred], axis=0
    ), axis=0).sum(-1)
    
    # find non-zero (exisitng gt label)
    inter = inter[torch.nonzero(union)]
    union = union[torch.nonzero(union)]
    return (inter / union).mean()

if __name__ == '__main__':
    gt = torch.tensor([0,1,2,5,6])
    test = torch.tensor([0,1,2,3,4])

    label_onehot = F.one_hot(gt, num_classes=80)
    pred_onehot = F.one_hot(test, num_classes=80)

    #This verify that the evaluation code is solid
    print(new_iou(pred_onehot, label_onehot))
    print(db_eval_multiclass_iou(gt, test))


    
