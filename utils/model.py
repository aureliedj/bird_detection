from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch  import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.models.detection.transform as T

def get_model_object_detection(num_classes = 2):
    
    model = fasterrcnn_resnet50_fpn(weights= FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_temp_model(num_classes = 2):
    
    model = fasterrcnn_resnet50_fpn(weights= FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    ## change first convolution to take 3*3 channels as inputs --> 9
    model.backbone.body.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    T=GeneralizedRCNNTransform(800, 1333, image_mean, image_std)
    model.transform = T

    ## check trainabe parameters
    #params = [p for p in model.parameters() if p.requires_grad]

    return model
    