from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np


object_dict = {
    0: 'background', 
    1: 'task', 
    2: 'exclusiveGateway', 
    3: 'event', 
    4: 'parallelGateway', 
    5: 'messageEvent', 
    6: 'pool', 
    7: 'lane', 
    8: 'dataObject', 
    9: 'dataStore', 
    10: 'subProcess', 
    11: 'eventBasedGateway', 
    12: 'timerEvent',
}

arrow_dict = {
    0: 'background', 
    1: 'sequenceFlow', 
    2: 'dataAssociation', 
    3: 'messageFlow', 
}

class_dict = {
    0: 'background', 
    1: 'task', 
    2: 'exclusiveGateway', 
    3: 'event', 
    4: 'parallelGateway', 
    5: 'messageEvent', 
    6: 'pool', 
    7: 'lane', 
    8: 'dataObject', 
    9: 'dataStore', 
    10: 'subProcess', 
    11: 'eventBasedGateway', 
    12: 'timerEvent',
    13: 'sequenceFlow', 
    14: 'dataAssociation', 
    15: 'messageFlow',
}


def is_vertical(box):
    """Determine if the text in the bounding box is vertically aligned."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (height > 2*width)

def rescale_boxes(scale, boxes):
    for i in range(len(boxes)):
                boxes[i] = [boxes[i][0]*scale,
                            boxes[i][1]*scale,
                            boxes[i][2]*scale,
                            boxes[i][3]*scale]
    return boxes

def iou(box1, box2):
    # Calcule l'intersection des deux boîtes englobantes
    inter_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    inter_area = max(0, inter_box[2] - inter_box[0]) * max(0, inter_box[3] - inter_box[1])

    # Calcule l'union des deux boîtes englobantes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def proportion_inside(box1, box2):
    # Calculate the intersection of the two bounding boxes
    inter_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    inter_area = max(0, inter_box[2] - inter_box[0]) * max(0, inter_box[3] - inter_box[1])

    # Calculate the area of box1
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # Calculate the proportion of box1 inside box2
    if box1_area == 0:
        return 0
    proportion = inter_area / box1_area

    # Ensure the proportion is at most 100%
    return min(proportion, 1.0)

def resize_boxes(boxes, original_size, target_size):
    """
    Resizes bounding boxes according to a new image size.

    Parameters:
    - boxes (np.array): The original bounding boxes as a numpy array of shape [N, 4].
    - original_size (tuple): The original size of the image as (width, height).
    - target_size (tuple): The desired size to resize the image to as (width, height).

    Returns:
    - np.array: The resized bounding boxes as a numpy array of shape [N, 4].
    """
    orig_width, orig_height = original_size
    target_width, target_height = target_size

    # Calculate the ratios for width and height
    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height

    # Apply the ratios to the bounding boxes
    boxes[:, 0] *= width_ratio
    boxes[:, 1] *= height_ratio
    boxes[:, 2] *= width_ratio
    boxes[:, 3] *= height_ratio

    return boxes

def resize_keypoints(keypoints: np.ndarray, original_size: tuple, target_size: tuple) -> np.ndarray:
    """
    Resize keypoints based on the original and target dimensions of an image.

    Parameters:
    - keypoints (np.ndarray): The array of keypoints, where each keypoint is represented by its (x, y) coordinates.
    - original_size (tuple): The width and height of the original image (width, height).
    - target_size (tuple): The width and height of the target image (width, height).

    Returns:
    - np.ndarray: The resized keypoints.

    Explanation:
    The function calculates the ratio of the target dimensions to the original dimensions.
    It then applies these ratios to the x and y coordinates of each keypoint to scale them
    appropriately to the target image size.
    """

    orig_width, orig_height = original_size
    target_width, target_height = target_size

    # Calculate the ratios for width and height scaling
    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height

    # Apply the scaling ratios to the x and y coordinates of each keypoint
    keypoints[:, 0] *= width_ratio  # Scale x coordinates
    keypoints[:, 1] *= height_ratio  # Scale y coordinates

    return keypoints



def find_other_keypoint(idx, keypoints, boxes):
    box = boxes[idx]
    key1,key2 = keypoints[idx]
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    average_keypoint = (key1 + key2) // 2
    #find the opposite keypoint to the center
    if average_keypoint[0] < center[0]:
        x = center[0] + abs(center[0] - average_keypoint[0])
    else:
        x = center[0] - abs(center[0] - average_keypoint[0])
    if average_keypoint[1] < center[1]:
        y = center[1] + abs(center[1] - average_keypoint[1])
    else:
        y = center[1] - abs(center[1] - average_keypoint[1])
    return x, y, average_keypoint[0], average_keypoint[1]
    

def filter_overlap_boxes(boxes, scores, labels, keypoints, iou_threshold=0.5):
    """
    Filters overlapping boxes based on the Intersection over Union (IoU) metric, keeping only the boxes with the highest scores.

    Parameters:
    - boxes (np.ndarray): Array of bounding boxes with shape (N, 4), where each row contains [x_min, y_min, x_max, y_max].
    - scores (np.ndarray): Array of scores for each box, reflecting the confidence of detection.
    - labels (np.ndarray): Array of labels corresponding to each box.
    - keypoints (np.ndarray): Array of keypoints associated with each box.
    - iou_threshold (float): Threshold for IoU above which a box is considered overlapping.

    Returns:
    - tuple: Filtered boxes, scores, labels, and keypoints.
    """
    # Calculate the area of each bounding box to use in IoU calculation.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort the indices of the boxes based on their scores in descending order.
    order = scores.argsort()[::-1]
    
    keep = []  # List to store indices of boxes to keep.
    
    while order.size > 0:
        # Take the first index (highest score) from the sorted list.
        i = order[0]
        keep.append(i)  # Add this index to 'keep' list.
        
        # Compute the coordinates of the intersection rectangle.
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        # Compute the area of the intersection rectangle.
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # Calculate IoU and find boxes with IoU less than the threshold to keep.
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        
        # Update the list of box indices to consider in the next iteration.
        order = order[inds + 1]  # Skip the first element since it's already included in 'keep'.
    
    # Use the indices in 'keep' to select the boxes, scores, labels, and keypoints to return.
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    keypoints = keypoints[keep]
    
    return boxes, scores, labels, keypoints



def find_closest_object(keypoint, boxes, labels):
    """
    Find the closest object to a keypoint based on their proximity.

    Parameters:
    - keypoint (numpy.ndarray): The coordinates of the keypoint.
    - boxes (numpy.ndarray): The bounding boxes of the objects.

    Returns:
    - int or None: The index of the closest object to the keypoint, or None if no object is found.
    """
    closest_object_idx = None
    best_point = None  
    min_distance = float('inf')
    # Iterate over each bounding box
    for i, box in enumerate(boxes):
        if labels[i] in [list(class_dict.values()).index('sequenceFlow'),
                         list(class_dict.values()).index('messageFlow'),
                         list(class_dict.values()).index('dataAssociation'),
                         #list(class_dict.values()).index('pool'),
                         list(class_dict.values()).index('lane')]:
            continue
        x1, y1, x2, y2 = box

        top = ((x1+x2)/2, y1)
        bottom = ((x1+x2)/2, y2)
        left = (x1, (y1+y2)/2)
        right = (x2, (y1+y2)/2)
        points = [left, top , right, bottom]

        pos_dict = {0:'left', 1:'top', 2:'right', 3:'bottom'}

        # Calculate the distance between the keypoint and the center of the bounding box
        for pos, (point) in enumerate(points):
            distance = np.linalg.norm(keypoint[:2] - point)
            # Update the closest object index if this object is closer
            if distance < min_distance:
                min_distance = distance
                closest_object_idx = i
                best_point = pos_dict[pos]

    return closest_object_idx, best_point


