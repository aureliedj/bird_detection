from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch  import nn

def get_model_object_detection(num_classes):

    model = fasterrcnn_resnet50_fpn_v2(weights= FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    ## change first convolution to take 3*3 channels as inputs --> 9
    # model.backbone.body.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    """## check trainabe parameters
    params = [p for p in model.parameters() if p.requires_grad]"""

    return model
    