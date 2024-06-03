from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
from torch.utils.data.dataloader import default_collate
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from torch.optim import SGD
import time
from torch.optim import AdamW
import copy
from torchvision import transforms


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


def iou(box1, box2):
    # Calcule l'intersection des deux boîtes englobantes
    inter_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    inter_area = max(0, inter_box[2] - inter_box[0]) * max(0, inter_box[3] - inter_box[1])

    # Calcule l'union des deux boîtes englobantes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

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



class RandomCrop:
    def __init__(self, new_size=(1333,800),crop_fraction=0.5, min_objects=4):
        self.crop_fraction = crop_fraction
        self.min_objects = min_objects
        self.new_size = new_size

    def __call__(self, image, target):
        new_w1, new_h1 = self.new_size
        w, h = image.size
        new_w = int(w * self.crop_fraction)
        new_h = int(new_w*new_h1/new_w1)

        i=0
        for i in range(4): 
          if new_h >= h:
            i += 0.05
            new_w = int(w * (self.crop_fraction - i))
            new_h = int(new_w*new_h1/new_w1)
          if new_h < h:
            continue

        if new_h >= h:
          return image, target

        boxes = target["boxes"]
        if 'keypoints' in target:
            keypoints = target["keypoints"]
        else:
            keypoints = []
            for i in range(len(boxes)):
                keypoints.append(torch.zeros((2,3)))
        

        # Attempt to find a suitable crop region
        success = False
        for _ in range(100):  # Max 100 attempts to find a valid crop
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            crop_region = [left, top, left + new_w, top + new_h]

            # Check how many objects are fully contained in this region
            contained_boxes = []
            contained_keypoints = []
            for box, kp in zip(boxes, keypoints):
                if box[0] >= crop_region[0] and box[1] >= crop_region[1] and box[2] <= crop_region[2] and box[3] <= crop_region[3]:
                    # Adjust box and keypoints coordinates
                    new_box = box - torch.tensor([crop_region[0], crop_region[1], crop_region[0], crop_region[1]])
                    new_kp = kp - torch.tensor([crop_region[0], crop_region[1], 0])
                    contained_boxes.append(new_box)
                    contained_keypoints.append(new_kp)

            if len(contained_boxes) >= self.min_objects:
                success = True
                break

        if success:
            # Perform the actual crop
            image = F.crop(image, top, left, new_h, new_w)
            target["boxes"] = torch.stack(contained_boxes) if contained_boxes else torch.zeros((0, 4))
            if 'keypoints' in target:
                target["keypoints"] = torch.stack(contained_keypoints) if contained_keypoints else torch.zeros((0, 2, 4))

        return image, target


class RandomFlip:
    def __init__(self, h_flip_prob=0.5, v_flip_prob=0.5):
        """
        Initializes the RandomFlip with probabilities for flipping.

        Parameters:
        - h_flip_prob (float): Probability of applying a horizontal flip to the image.
        - v_flip_prob (float): Probability of applying a vertical flip to the image.
        """
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def __call__(self, image, target):
        """
        Applies random horizontal and/or vertical flip to the image and updates target data accordingly.

        Parameters:
        - image (PIL Image): The image to be flipped.
        - target (dict): The target dictionary containing 'boxes' and 'keypoints'.

        Returns:
        - PIL Image, dict: The flipped image and its updated target dictionary.
        """
        if random.random() < self.h_flip_prob:
            image = F.hflip(image)
            w, _ = image.size  # Get the new width of the image after flip for bounding box adjustment
            # Adjust bounding boxes for horizontal flip
            for i, box in enumerate(target['boxes']):
                xmin, ymin, xmax, ymax = box
                target['boxes'][i] = torch.tensor([w - xmax, ymin, w - xmin, ymax], dtype=torch.float32)

            # Adjust keypoints for horizontal flip
            if 'keypoints' in target:
                new_keypoints = []
                for keypoints_for_object in target['keypoints']:
                    flipped_keypoints_for_object = []
                    for kp in keypoints_for_object:
                        x, y = kp[:2]
                        new_x = w - x
                        flipped_keypoints_for_object.append(torch.tensor([new_x, y] + list(kp[2:])))
                    new_keypoints.append(torch.stack(flipped_keypoints_for_object))
                target['keypoints'] = torch.stack(new_keypoints)

        if random.random() < self.v_flip_prob:
            image = F.vflip(image)
            _, h = image.size  # Get the new height of the image after flip for bounding box adjustment
            # Adjust bounding boxes for vertical flip
            for i, box in enumerate(target['boxes']):
                xmin, ymin, xmax, ymax = box
                target['boxes'][i] = torch.tensor([xmin, h - ymax, xmax, h - ymin], dtype=torch.float32)

            # Adjust keypoints for vertical flip
            if 'keypoints' in target:
                new_keypoints = []
                for keypoints_for_object in target['keypoints']:
                    flipped_keypoints_for_object = []
                    for kp in keypoints_for_object:
                        x, y = kp[:2]
                        new_y = h - y
                        flipped_keypoints_for_object.append(torch.tensor([x, new_y] + list(kp[2:])))
                    new_keypoints.append(torch.stack(flipped_keypoints_for_object))
                target['keypoints'] = torch.stack(new_keypoints)

        return image, target
    

class RandomRotate:
    def __init__(self, max_rotate_deg=20, rotate_proba=0.3):
        """
        Initializes the RandomRotate with a maximum rotation angle and probability of rotating.

        Parameters:
        - max_rotate_deg (int): Maximum degree to rotate the image.
        - rotate_proba (float): Probability of applying rotation to the image.
        """
        self.max_rotate_deg = max_rotate_deg
        self.rotate_proba = rotate_proba

    def __call__(self, image, target):
        """
        Randomly rotates the image and updates the target data accordingly.

        Parameters:
        - image (PIL Image): The image to be rotated.
        - target (dict): The target dictionary containing 'boxes', 'labels', and 'keypoints'.

        Returns:
        - PIL Image, dict: The rotated image and its updated target dictionary.
        """
        if random.random() < self.rotate_proba:
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            image = F.rotate(image, angle, expand=False, fill=200)

            # Rotate bounding boxes
            w, h = image.size
            cx, cy = w / 2, h / 2
            boxes = target["boxes"]
            new_boxes = []
            for box in boxes:
                new_box = self.rotate_box(box, angle, cx, cy)
                new_boxes.append(new_box)
            target["boxes"] = torch.stack(new_boxes)

            # Rotate keypoints
            if 'keypoints' in target:
                new_keypoints = []
                for keypoints in target["keypoints"]:
                    new_kp = self.rotate_keypoints(keypoints, angle, cx, cy)
                    new_keypoints.append(new_kp)
                target["keypoints"] = torch.stack(new_keypoints)

        return image, target

    def rotate_box(self, box, angle, cx, cy):
        """
        Rotates a bounding box by a given angle around the center of the image.
        """
        x1, y1, x2, y2 = box
        corners = torch.tensor([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim=1)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        corners = torch.matmul(torch.tensor(M, dtype=torch.float32), corners.T).T
        x_ = corners[:, 0]
        y_ = corners[:, 1]
        x_min, x_max = torch.min(x_), torch.max(x_)
        y_min, y_max = torch.min(y_), torch.max(y_)
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

    def rotate_keypoints(self, keypoints, angle, cx, cy):
        """
        Rotates keypoints by a given angle around the center of the image.
        """
        new_keypoints = []
        for kp in keypoints:
            x, y, v = kp
            point = torch.tensor([x, y, 1])
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            new_point = torch.matmul(torch.tensor(M, dtype=torch.float32), point)
            new_keypoints.append(torch.tensor([new_point[0], new_point[1], v], dtype=torch.float32))
        return torch.stack(new_keypoints)

def rotate_90_box(box, angle, w, h):
    x1, y1, x2, y2 = box
    if angle == 90:
        return torch.tensor([y1,h-x2,y2,h-x1])
    elif angle == 270 or angle == -90:
        return torch.tensor([w-y2,x1,w-y1,x2])
    else:
        print("angle not supported")

def rotate_90_keypoints(kp, angle, w, h):
    # Extract coordinates and visibility from each keypoint tensor
    x1, y1, v1 = kp[0][0], kp[0][1], kp[0][2]
    x2, y2, v2 = kp[1][0], kp[1][1], kp[1][2]
    # Swap x and y coordinates for each keypoint
    if angle == 90:
        new = [[y1, h-x1, v1], [y2, h-x2, v2]]
    elif angle == 270 or angle == -90:
        new = [[w-y1, x1, v1], [w-y2, x2, v2]]

    return torch.tensor(new, dtype=torch.float32)
    

def rotate_vertical(image, target):
    # Rotate the image and target if the image is vertical
    new_boxes = []
    angle = random.choice([-90,90])
    image = F.rotate(image, angle, expand=True, fill=200)
    for box in target["boxes"]:
        new_box = rotate_90_box(box, angle, image.size[0], image.size[1])
        new_boxes.append(new_box)
    target["boxes"] = torch.stack(new_boxes)
    
    if 'keypoints' in target:
        new_kp = []  
        for kp in target['keypoints']:                   
            new_key = rotate_90_keypoints(kp, angle, image.size[0], image.size[1])
            new_kp.append(new_key)
        target['keypoints'] = torch.stack(new_kp)
    return image, target

class BPMN_Dataset(Dataset):
    def __init__(self, annotations, transform=None, crop_transform=None, crop_prob=0.3, rotate_90_proba=0.2, flip_transform=None, rotate_transform=None, new_size=(1333,800),keep_ratio=False,resize=True, model_type='object', rotate_vertical=False):
        self.annotations = annotations
        print(f"Loaded {len(self.annotations)} annotations.")
        self.transform = transform
        self.crop_transform = crop_transform
        self.crop_prob = crop_prob
        self.flip_transform = flip_transform
        self.rotate_transform = rotate_transform
        self.resize = resize
        self.rotate_vertical = rotate_vertical
        self.new_size = new_size
        self.keep_ratio = keep_ratio
        self.model_type = model_type
        if model_type == 'object':
            self.dict = object_dict
        elif model_type == 'arrow':
            self.dict = arrow_dict
        self.rotate_90_proba = rotate_90_proba

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image = annotation.img.convert("RGB")
        boxes = torch.tensor(np.array(annotation.boxes_ltrb), dtype=torch.float32)
        labels_names = [ann for ann in annotation.categories]

        #only keep the labels, boxes and keypoints that are in the class_dict
        kept_indices = [i for i, ann in enumerate(annotation.categories) if ann in self.dict.values()]
        boxes = boxes[kept_indices]
        labels_names = [ann for i, ann in enumerate(labels_names) if i in kept_indices]

        labels_id = torch.tensor([(list(self.dict.values()).index(ann)) for ann in labels_names], dtype=torch.int64)

        # Initialize keypoints tensor
        max_keypoints = 2
        keypoints = torch.zeros((len(labels_id), max_keypoints, 3), dtype=torch.float32)

        ii=0
        for i, ann in enumerate(annotation.annotations):
            #only keep the keypoints that are in the kept indices
            if i not in kept_indices:
                continue
            if ann.category in ["sequenceFlow", "messageFlow", "dataAssociation"]:
                # Fill the keypoints tensor for this annotation, mark as visible (1)
                kp = np.array(ann.keypoints, dtype=np.float32).reshape(-1, 3)
                kp = kp[:,:2]
                visible = np.ones((kp.shape[0], 1), dtype=np.float32)
                kp = np.hstack([kp, visible])
                keypoints[ii, :kp.shape[0], :] = torch.tensor(kp, dtype=torch.float32)
                ii += 1

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if self.model_type == 'object':        
            target = {
                "boxes": boxes,
                "labels": labels_id,
                #"area": area,
                #"keypoints": keypoints,
            }
        elif self.model_type == 'arrow':
            target = {
                "boxes": boxes,
                "labels": labels_id,
                #"area": area,
                "keypoints": keypoints,
            }

        # Randomly apply flip transform
        if self.flip_transform:
            image, target = self.flip_transform(image, target)

        # Randomly apply rotate transform
        if self.rotate_transform:
            image, target = self.rotate_transform(image, target)

        # Randomly apply the custom cropping transform
        if self.crop_transform and random.random() < self.crop_prob:
            image, target = self.crop_transform(image, target)
            
        # Rotate vertical image
        if self.rotate_vertical and random.random() < self.rotate_90_proba:
            image, target = rotate_vertical(image, target)

        if self.resize:
            if self.keep_ratio:
                original_size = image.size
                # Calculate scale to fit the new size while maintaining aspect ratio
                scale = min(self.new_size[0] / original_size[0], self.new_size[1] / original_size[1])
                new_scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))

                target['boxes'] = resize_boxes(target['boxes'], (image.size[0],image.size[1]), (new_scaled_size))
                if 'area' in target:
                    target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])

                if 'keypoints' in target:
                    for i in range(len(target['keypoints'])):
                        target['keypoints'][i] = resize_keypoints(target['keypoints'][i], (image.size[0],image.size[1]), (new_scaled_size))

                # Resize image to new scaled size
                image = F.resize(image, (new_scaled_size[1], new_scaled_size[0]))

                # Pad the resized image to make it exactly the desired size
                padding = [0, 0, self.new_size[0] - new_scaled_size[0], self.new_size[1] - new_scaled_size[1]]
                image = F.pad(image, padding, fill=200, padding_mode='constant')
            else:
                target['boxes'] = resize_boxes(target['boxes'], (image.size[0],image.size[1]), self.new_size)
                if 'area' in target:
                    target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
                if 'keypoints' in target:
                    for i in range(len(target['keypoints'])):
                        target['keypoints'][i] = resize_keypoints(target['keypoints'][i], (image.size[0],image.size[1]), self.new_size)
                image = F.resize(image, (self.new_size[1], self.new_size[0]))

        return self.transform(image), target

def collate_fn(batch):
    """
    Custom collation function for DataLoader that handles batches of images and targets.

    This function ensures that images are properly batched together using PyTorch's default collation,
    while keeping the targets (such as bounding boxes and labels) in a list of dictionaries, 
    as each image might have a different number of objects detected.

    Parameters:
    - batch (list): A list of tuples, where each tuple contains an image and its corresponding target dictionary.

    Returns:
    - Tuple containing:
      - Tensor: Batched images.
      - List of dicts: Targets corresponding to each image in the batch.
    """
    images, targets = zip(*batch)  # Unzip the batch into separate lists for images and targets.

    # Batch images using the default collate function which handles tensors, numpy arrays, numbers, etc.
    images = default_collate(images)

    return images, targets



def create_loader(new_size,transformation, annotations1, annotations2=None, 
                  batch_size=4, crop_prob=0.2, crop_fraction=0.7, min_objects=3, 
                  h_flip_prob=0.3, v_flip_prob=0.3, max_rotate_deg=20, rotate_90_proba=0.2, rotate_proba=0.3, 
                  seed=42, resize=True, rotate_vertical=False, keep_ratio=False, model_type = 'object'):
    """
    Creates a DataLoader for BPMN datasets with optional transformations and concatenation of two datasets.

    Parameters:
    - transformation (callable): Transformation function to apply to each image (e.g., normalization).
    - annotations1 (list): Primary list of annotations.
    - annotations2 (list, optional): Secondary list of annotations to concatenate with the first.
    - batch_size (int): Number of images per batch.
    - crop_prob (float): Probability of applying the crop transformation.
    - crop_fraction (float): Fraction of the original width to use when cropping.
    - min_objects (int): Minimum number of objects required to be within the crop.
    - h_flip_prob (float): Probability of applying horizontal flip.
    - v_flip_prob (float): Probability of applying vertical flip.
    - seed (int): Seed for random number generators for reproducibility.
    - resize (bool): Flag indicating whether to resize images after transformations.

    Returns:
    - DataLoader: Configured data loader for the dataset.
    """

    # Initialize custom transformations for cropping and flipping
    custom_crop_transform = RandomCrop(new_size,crop_fraction, min_objects)
    custom_flip_transform = RandomFlip(h_flip_prob, v_flip_prob)
    custom_rotate_transform = RandomRotate(max_rotate_deg, rotate_proba)

    # Create the primary dataset
    dataset = BPMN_Dataset(
        annotations=annotations1,
        transform=transformation,
        crop_transform=custom_crop_transform,
        crop_prob=crop_prob,
        rotate_90_proba=rotate_90_proba,
        flip_transform=custom_flip_transform,
        rotate_transform=custom_rotate_transform,
        rotate_vertical=rotate_vertical,
        new_size=new_size,
        keep_ratio=keep_ratio,
        model_type=model_type,
        resize=resize
    )

    # Optionally concatenate a second dataset
    if annotations2:
        dataset2 = BPMN_Dataset(
            annotations=annotations2,
            transform=transformation,
            crop_transform=custom_crop_transform,
            crop_prob=crop_prob,
            rotate_90_proba=rotate_90_proba,
            flip_transform=custom_flip_transform,
            rotate_vertical=rotate_vertical,
            new_size=new_size,
            keep_ratio=keep_ratio,
            model_type=model_type,
            resize=resize
        )
        dataset = ConcatDataset([dataset, dataset2])  # Concatenate the two datasets

    # Set the seed for reproducibility in random operations within transformations and data loading
    random.seed(seed)
    torch.manual_seed(seed)

    # Create the DataLoader with the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return data_loader



def write_results(name_model,metrics_list,start_epoch):
  with open('./results/'+ name_model+ '.txt', 'w') as f:
        for i in range(len(metrics_list[0])):
          f.write(f"{i+1+start_epoch},{metrics_list[0][i]},{metrics_list[1][i]},{metrics_list[2][i]},{metrics_list[3][i]},{metrics_list[4][i]},{metrics_list[5][i]},{metrics_list[6][i]},{metrics_list[7][i]},{metrics_list[8][i]},{metrics_list[9][i]} \n")


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



def draw_annotations(image, 
                     target=None, 
                     prediction=None, 
                     full_prediction=None,
                     text_predictions=None, 
                     model_dict=class_dict, 
                     draw_keypoints=False, 
                     draw_boxes=False, 
                     draw_text=False,
                     draw_links=False,
                     draw_twins=False,
                     write_class=False,
                     write_score=False, 
                     write_text=False,
                     write_idx=False,
                     score_threshold=0.4, 
                     keypoints_correction=False,
                     only_print=None,
                     axis=False,
                     return_image=False,
                     new_size=(1333,800),
                     resize=False):
    """
    Draws annotations on images including bounding boxes, keypoints, links, and text.
    
    Parameters:
    - image (np.array): The image on which annotations will be drawn.
    - target (dict): Ground truth data containing boxes, labels, etc.
    - prediction (dict): Prediction data from a model.
    - full_prediction (dict): Additional detailed prediction data, potentially including relationships.
    - text_predictions (tuple): OCR text predictions containing bounding boxes and texts.
    - model_dict (dict): Mapping from class IDs to class names.
    - draw_keypoints (bool): Flag to draw keypoints.
    - draw_boxes (bool): Flag to draw bounding boxes.
    - draw_text (bool): Flag to draw text annotations.
    - draw_links (bool): Flag to draw links between annotations.
    - draw_twins (bool): Flag to draw twins keypoints.
    - write_class (bool): Flag to write class names near the annotations.
    - write_score (bool): Flag to write scores near the annotations.
    - write_text (bool): Flag to write OCR recognized text.
    - score_threshold (float): Threshold for scores above which annotations will be drawn.
    - only_print (str): Specific class name to filter annotations by.
    - resize (bool): Whether to resize annotations to fit the image size.
    """

    # Convert image to RGB (if not already in that format)
    if prediction is None:
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    scale = max(image.shape[0], image.shape[1]) / 1000

    # Function to draw bounding boxes and keypoints
    def draw(data,is_prediction=False):
        """ Helper function to draw annotations based on provided data. """

        for i in range(len(data['boxes'])):
            if is_prediction:
                box = data['boxes'][i].tolist()
                x1, y1, x2, y2 = box
                if resize:
                    x1, y1, x2, y2 = resize_boxes(np.array([box]), new_size, (image_copy.shape[1],image_copy.shape[0]))[0]
                score = data['scores'][i].item()
                if score < score_threshold:
                    continue
            else:
                box = data['boxes'][i].tolist()
                x1, y1, x2, y2 = box
            if draw_boxes:
                if only_print is not None:
                    if data['labels'][i] != list(model_dict.values()).index(only_print):
                        continue
                cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0) if is_prediction else (0, 0, 0), int(2*scale))
            if is_prediction and write_score:
                cv2.putText(image_copy, str(round(score, 2)), (int(x1), int(y1) + int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (100,100, 255), 2)

            if write_class and 'labels' in data:
                class_id = data['labels'][i].item()
                cv2.putText(image_copy, model_dict[class_id], (int(x1), int(y1) - int(2*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (255, 100, 100), 2)

            if write_idx:
                cv2.putText(image_copy, str(i), (int(x1) + int(15*scale), int(y1) + int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 2*scale, (0,0, 0), 2)
  

            # Draw keypoints if available
            if draw_keypoints and 'keypoints' in data:
                if is_prediction and keypoints_correction:
                    for idx, (key1, key2) in enumerate(data['keypoints']):
                        if data['labels'][idx] not in [list(model_dict.values()).index('sequenceFlow'),
                                    list(model_dict.values()).index('messageFlow'),
                                    list(model_dict.values()).index('dataAssociation')]:
                            continue
                        # Calculate the Euclidean distance between the two keypoints
                        distance = np.linalg.norm(key1[:2] - key2[:2])
                
                        if distance < 5:
                            x_new,y_new, x,y = find_other_keypoint(idx, data['keypoints'], data['boxes'])
                            data['keypoints'][idx][0] = torch.tensor([x_new, y_new,1])
                            data['keypoints'][idx][1] = torch.tensor([x, y,1])
                            print("keypoint has been changed")
                for i in range(len(data['keypoints'])):
                    kp = data['keypoints'][i]
                    for j in range(kp.shape[0]):
                        if is_prediction and data['labels'][i] != list(model_dict.values()).index('sequenceFlow') and data['labels'][i] != list(model_dict.values()).index('messageFlow') and data['labels'][i] != list(model_dict.values()).index('dataAssociation'):
                            continue
                        if is_prediction:
                            score = data['scores'][i]
                            if score < score_threshold:
                                continue
                        x,y,v = np.array(kp[j])
                        if resize:
                            x, y, v = resize_keypoints(np.array([kp[j]]), new_size, (image_copy.shape[1],image_copy.shape[0]))[0]
                        if j == 0:
                            cv2.circle(image_copy, (int(x), int(y)), int(5*scale), (0, 0, 255), -1)
                        else:
                            cv2.circle(image_copy, (int(x), int(y)), int(5*scale), (255, 0, 0), -1)

        # Draw text predictions if available
        if (draw_text or write_text) and text_predictions is not None:                        
            for i in range(len(text_predictions[0])):
                x1, y1, x2, y2 = text_predictions[0][i]
                text = text_predictions[1][i]
                if resize:
                    x1, y1, x2, y2 = resize_boxes(np.array([[float(x1), float(y1), float(x2), float(y2)]]), new_size, (image_copy.shape[1],image_copy.shape[0]))[0]
                if draw_text:
                    cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), int(2*scale))
                if write_text:
                    cv2.putText(image_copy, text, (int(x1 + int(2*scale)), int((y1+y2)/2) ), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (0,0, 0), 2)
            
    def draw_with_links(full_prediction):
        '''Draws links between objects based on the full prediction data.'''
        #check if keypoints detected are the same
        if draw_twins and full_prediction is not None:
            # Pre-calculate indices for performance
            circle_color = (0, 255, 0)  # Green color for the circle
            circle_radius = int(10 * scale)  # Circle radius scaled by image scale

            for idx, (key1, key2) in enumerate(full_prediction['keypoints']):
                if full_prediction['labels'][idx] not in [list(model_dict.values()).index('sequenceFlow'),
                         list(model_dict.values()).index('messageFlow'),
                         list(model_dict.values()).index('dataAssociation')]:
                    continue
                # Calculate the Euclidean distance between the two keypoints
                distance = np.linalg.norm(key1[:2] - key2[:2])
                if distance < 10:
                    x_new,y_new, x,y = find_other_keypoint(idx,full_prediction)
                    cv2.circle(image_copy, (int(x), int(y)), circle_radius, circle_color, -1)
                    cv2.circle(image_copy, (int(x_new), int(y_new)), circle_radius, (0,0,0), -1)

        # Draw links between objects
        if draw_links==True and full_prediction is not None:
            for i, (start_idx, end_idx) in enumerate(full_prediction['links']):
                if start_idx is None or end_idx is None:
                    continue
                start_box = full_prediction['boxes'][start_idx]
                end_box = full_prediction['boxes'][end_idx]
                current_box = full_prediction['boxes'][i]
                # Calculate the center of each bounding box
                start_center = ((start_box[0] + start_box[2]) // 2, (start_box[1] + start_box[3]) // 2)
                end_center = ((end_box[0] + end_box[2]) // 2, (end_box[1] + end_box[3]) // 2)
                current_center = ((current_box[0] + current_box[2]) // 2, (current_box[1] + current_box[3]) // 2)
                # Draw a line between the centers of the connected objects
                cv2.line(image_copy, (int(start_center[0]), int(start_center[1])), (int(current_center[0]), int(current_center[1])), (0, 0, 255), int(2*scale))
                cv2.line(image_copy, (int(current_center[0]), int(current_center[1])), (int(end_center[0]), int(end_center[1])), (255, 0, 0), int(2*scale))

                i+=1

    # Draw GT annotations
    if target is not None:
        draw(target, is_prediction=False)
    # Draw predictions
    if prediction is not None:
        #prediction = prediction[0] 
        draw(prediction, is_prediction=True)
    # Draw links with full predictions
    if full_prediction is not None:
        draw_with_links(full_prediction)

    # Display the image
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(image_copy)
    if axis==False:
        plt.axis('off')
    plt.show()

    if return_image:
        return image_copy

def find_closest_object(keypoint, boxes, labels):
    """
    Find the closest object to a keypoint based on their proximity.

    Parameters:
    - keypoint (numpy.ndarray): The coordinates of the keypoint.
    - boxes (numpy.ndarray): The bounding boxes of the objects.

    Returns:
    - int or None: The index of the closest object to the keypoint, or None if no object is found.
    """
    min_distance = float('inf')
    closest_object_idx = None
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

        # Calculate the distance between the keypoint and the center of the bounding box
        for point in points:
            distance = np.linalg.norm(keypoint[:2] - point)
            # Update the closest object index if this object is closer
            if distance < min_distance:
                min_distance = distance
                closest_object_idx = i
                best_point = point

    return closest_object_idx, best_point

