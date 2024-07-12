import copy
import numpy as np
import random
import time
import torch
import torchvision.transforms.functional as F

from modules.eval import main_evaluation
from torch.optim import SGD, AdamW
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights




def get_arrow_model(num_classes, num_keypoints=2):
    """
    Configures and returns a modified Keypoint R-CNN model based on ResNet-50 with FPN, adapted for a custom number of classes and keypoints.

    Parameters:
    - num_classes (int): Number of classes for the model to detect, excluding the background class.
    - num_keypoints (int): Number of keypoints to predict for each detected object.

    Returns:
    - model (torch.nn.Module): The modified Keypoint R-CNN model.

    Steps:
    1. Load a pre-trained Keypoint R-CNN model with a ResNet-50 backbone and Feature Pyramid Network (FPN). 
       The model is initially configured for the COCO dataset, which includes various object classes and keypoints.
    2. Replace the box predictor to adjust the number of output classes. The box predictor is responsible for
       classifying detected regions and predicting their bounding boxes.
    3. Replace the keypoint predictor to adjust the number of keypoints the model predicts for each object.
       This is necessary to tailor the model to specific tasks that may have different keypoint structures.
    """

    model = keypointrcnn_resnet50_fpn(weights=None)

    # Get the number of input features for the classifier in the box predictor.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the box predictor in the ROI heads with a new one, tailored to the number of classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the keypoint predictor in the ROI heads with a new one, specifically designed for the desired number of keypoints.
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)

    return model


def get_faster_rcnn_model(num_classes):
    """
    Configures and returns a modified Faster R-CNN model based on ResNet-50 with FPN, adapted for a custom number of classes.

    Parameters:
    - num_classes (int): Number of classes for the model to detect, including the background class.

    Returns:
    - model (torch.nn.Module): The modified Faster R-CNN model.
    """
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=None)

    # Get the number of input features for the classifier in the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the box predictor with a new one, tailored to the number of classes (num_classes includes the background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def prepare_model(dict,opti,learning_rate= 0.0003,model_to_load=None, model_type = 'object'):
  # Adjusted to pass the class_dict directly
  if model_type == 'object':
    model = get_faster_rcnn_model(len(dict))
  elif model_type == 'arrow':
    model = get_arrow_model(len(dict),2)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  # Load the model weights
  if model_to_load:
    model.load_state_dict(torch.load('./models/'+ model_to_load +'.pth', map_location=device))
    print(f"Model '{model_to_load}'  loaded")

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  if opti == 'SGD':
    #learning_rate= 0.002
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
  elif opti == 'Adam':
    #learning_rate = 0.0003
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00056, eps=1e-08, betas=(0.9, 0.999))
  else:
    print('Optimizer not found')

  return model, optimizer, device

