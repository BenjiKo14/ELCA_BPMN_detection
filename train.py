import copy
import cv2
import numpy as np
import random
import time
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from eval import main_evaluation
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from tqdm import tqdm
from utils import write_results




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
    # Load a model pre-trained on COCO, initialized without pre-trained weights
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device == torch.device('cuda'):
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    else:
        model = keypointrcnn_resnet50_fpn(weights=False)

    # Get the number of input features for the classifier in the box predictor.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the box predictor in the ROI heads with a new one, tailored to the number of classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the keypoint predictor in the ROI heads with a new one, specifically designed for the desired number of keypoints.
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)

    return model

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
def get_faster_rcnn_model(num_classes):
    """
    Configures and returns a modified Faster R-CNN model based on ResNet-50 with FPN, adapted for a custom number of classes.

    Parameters:
    - num_classes (int): Number of classes for the model to detect, including the background class.

    Returns:
    - model (torch.nn.Module): The modified Faster R-CNN model.
    """
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

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




def evaluate_loss(model, data_loader, device, loss_config=None, print_losses=False):
    model.train()  # Set the model to evaluation mode
    total_loss = 0

    # Initialize lists to keep track of individual losses
    loss_classifier_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_box_reg_list = []
    loss_keypoints_list = []

    with torch.no_grad():  # Disable gradient computation
        for images, targets_im in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets_im]

            loss_dict = model(images, targets)

            # Calculate the total loss for the current batch
            losses = 0
            if loss_config is not None:
                for key, loss in loss_dict.items():
                    if loss_config.get(key, False):
                        losses += loss
            else:
                losses = sum(loss for key, loss in loss_dict.items())

            total_loss += losses.item()

            # Collect individual losses
            if loss_dict.get('loss_classifier') is not None:
                loss_classifier_list.append(loss_dict['loss_classifier'].item())
            else:
                loss_classifier_list.append(0)

            if loss_dict.get('loss_box_reg') is not None:
                loss_box_reg_list.append(loss_dict['loss_box_reg'].item())
            else:
                loss_box_reg_list.append(0)

            if loss_dict.get('loss_objectness') is not None:
                loss_objectness_list.append(loss_dict['loss_objectness'].item())
            else:
                loss_objectness_list.append(0)

            if loss_dict.get('loss_rpn_box_reg') is not None:
                loss_rpn_box_reg_list.append(loss_dict['loss_rpn_box_reg'].item())
            else:
                loss_rpn_box_reg_list.append(0)

            if 'loss_keypoint' in loss_dict:
                loss_keypoints_list.append(loss_dict['loss_keypoint'].item())
            else:
                loss_keypoints_list.append(0)

    # Calculate average loss
    avg_loss = total_loss / len(data_loader)

    avg_loss_classifier = np.mean(loss_classifier_list)
    avg_loss_box_reg = np.mean(loss_box_reg_list)
    avg_loss_objectness = np.mean(loss_objectness_list)
    avg_loss_rpn_box_reg = np.mean(loss_rpn_box_reg_list)
    avg_loss_keypoints = np.mean(loss_keypoints_list)

    if print_losses:
      print(f"Average Loss: {avg_loss:.4f}")
      print(f"Average Classifier Loss: {avg_loss_classifier:.4f}")
      print(f"Average Box Regression Loss: {avg_loss_box_reg:.4f}")
      print(f"Average Objectness Loss: {avg_loss_objectness:.4f}")
      print(f"Average RPN Box Regression Loss: {avg_loss_rpn_box_reg:.4f}")
      print(f"Average Keypoints Loss: {avg_loss_keypoints:.4f}")

    return avg_loss


def training_model(num_epochs, model, data_loader, subset_test_loader,
                   optimizer, model_to_load=None, change_learning_rate=5, start_key=30,
                   batch_size=4, crop_prob=0.2, h_flip_prob=0.3, v_flip_prob=0.3,
                   max_rotate_deg=20, rotate_proba=0.2, blur_prob=0.2,
                   score_threshold=0.7, iou_threshold=0.5, early_stop_f1_score=0.97,
                   information_training='training', start_epoch=0, loss_config=None, model_type = 'object',
                   eval_metric='f1_score', device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):


  if loss_config is None:
     print('No loss config found, all losses will be used.')
  else:
     #print the list of the losses that will be used
      print('The following losses will be used: ', end='')
      for key, value in loss_config.items():
          if value:
              print(key, end=", ")
      print()


  # Initialize lists to store epoch-wise average losses
  epoch_avg_losses = []
  epoch_avg_loss_classifier = []
  epoch_avg_loss_box_reg = []
  epoch_avg_loss_objectness = []
  epoch_avg_loss_rpn_box_reg = []
  epoch_avg_loss_keypoints = []
  epoch_precision = []
  epoch_recall = []
  epoch_f1_score = []
  epoch_test_loss = []


  start_tot = time.time()
  best_metrics = -1000
  best_epoch = 0
  best_model_state = None
  same = 0
  learning_rate = optimizer.param_groups[0]['lr']
  bad_test_loss = 0
  previous_test_loss = 1000

  print(f"Let's go training {model_type} model with {num_epochs} epochs!")
  print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Crop prob: {crop_prob}, Flip prob: {h_flip_prob}, Rotate prob: {rotate_proba}, Blur prob: {blur_prob}")

  for epoch in range(num_epochs):

      if (epoch>0 and (epoch)%change_learning_rate == 0) or bad_test_loss>1:
        learning_rate = 0.7*learning_rate
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=learning_rate, eps=1e-08, betas=(0.9, 0.999))
        print(f'Learning rate changed to {learning_rate:.4} and the best epoch for now is {best_epoch}')
        bad_test_loss = 0
      if epoch>0 and (epoch)==start_key:
        print("Now it's training Keypoints also")
        loss_config['loss_keypoint'] = True
        for name, param in model.named_parameters():
          if 'keypoint' in name:
              param.requires_grad = True

      model.train()
      start = time.time()
      total_loss = 0

      # Initialize lists to keep track of individual losses
      loss_classifier_list = []
      loss_box_reg_list = []
      loss_objectness_list = []
      loss_rpn_box_reg_list = []
      loss_keypoints_list =  []

      # Create a tqdm progress bar
      progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1+start_epoch}')

      for images, targets_im in progress_bar:
          images = [image.to(device) for image in images]
          targets = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets_im]

          optimizer.zero_grad()

          loss_dict = model(images, targets)
          # Inside the training loop where losses are calculated:
          losses = 0
          if loss_config is not None:
            for key, loss in loss_dict.items():
                if loss_config.get(key, False):
                    if key == 'loss_classifier':
                      loss *= 3
                    losses += loss
          else:
            losses = sum(loss for key, loss in loss_dict.items())

          # Collect individual losses
          if loss_dict['loss_classifier']:
            loss_classifier_list.append(loss_dict['loss_classifier'].item())
          else:
            loss_classifier_list.append(0)

          if loss_dict['loss_box_reg']:
            loss_box_reg_list.append(loss_dict['loss_box_reg'].item())
          else:
            loss_box_reg_list.append(0)

          if loss_dict['loss_objectness']:
            loss_objectness_list.append(loss_dict['loss_objectness'].item())
          else:
            loss_objectness_list.append(0)

          if loss_dict['loss_rpn_box_reg']:
            loss_rpn_box_reg_list.append(loss_dict['loss_rpn_box_reg'].item())
          else:
            loss_rpn_box_reg_list.append(0)

          if 'loss_keypoint' in loss_dict:
            loss_keypoints_list.append(loss_dict['loss_keypoint'].item())
          else:
            loss_keypoints_list.append(0)


          losses.backward()
          optimizer.step()

          total_loss += losses.item()

          # Update the description with the current loss
          progress_bar.set_description(f'Epoch {epoch+1+start_epoch}, Loss: {losses.item():.4f}')

      # Calculate average loss
      avg_loss = total_loss / len(data_loader)

      epoch_avg_losses.append(avg_loss)
      epoch_avg_loss_classifier.append(np.mean(loss_classifier_list))
      epoch_avg_loss_box_reg.append(np.mean(loss_box_reg_list))
      epoch_avg_loss_objectness.append(np.mean(loss_objectness_list))
      epoch_avg_loss_rpn_box_reg.append(np.mean(loss_rpn_box_reg_list))
      epoch_avg_loss_keypoints.append(np.mean(loss_keypoints_list))


        # Evaluate the model on the test set
      if eval_metric != 'loss':
        avg_test_loss = 0
        labels_precision, precision, recall, f1_score, key_accuracy, reverted_accuracy = main_evaluation(model, subset_test_loader,score_threshold=0.5, iou_threshold=0.5, distance_threshold=10, key_correction=False, model_type=model_type)
        print(f"Epoch {epoch+1+start_epoch}, Average Loss: {avg_loss:.4f}, Labels_precision: {labels_precision:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f} ", end=", ")
        if eval_metric == 'all':
          avg_test_loss = evaluate_loss(model, subset_test_loader, device, loss_config)
          print(f"Epoch {epoch+1+start_epoch}, Average Test Loss: {avg_test_loss:.4f}", end=", ")
      if eval_metric == 'loss':
        labels_precision, precision, recall, f1_score, key_accuracy, reverted_accuracy = 0,0,0,0,0,0
        avg_test_loss = evaluate_loss(model, subset_test_loader, device, loss_config)
        print(f"Epoch {epoch+1+start_epoch}, Average Training Loss: {avg_loss:.4f}, Average Test Loss: {avg_test_loss:.4f}", end=", ")

      print(f"Time: {time.time() - start:.2f} [s]")


      if epoch>0 and (epoch)%start_key == 0:
        print(f"Keypoints Accuracy: {key_accuracy:.4f}", end=", ")

      if eval_metric == 'f1_score':
        metric_used = f1_score
      elif eval_metric == 'precision':
        metric_used = precision
      elif eval_metric == 'recall':
        metric_used = recall
      else:
        metric_used = -avg_test_loss

      # Check if this epoch's model has the lowest average loss
      if metric_used > best_metrics:
          best_metrics = metric_used
          best_epoch = epoch+1+start_epoch
          best_model_state = copy.deepcopy(model.state_dict())

      if epoch>0 and f1_score>early_stop_f1_score:
        same+=1

      epoch_precision.append(precision)
      epoch_recall.append(recall)
      epoch_f1_score.append(f1_score)
      epoch_test_loss.append(avg_test_loss)

      name_model = f"model_{type(optimizer).__name__}_{epoch+1+start_epoch}ep_{batch_size}batch_trainval_blur0{int(blur_prob*10)}_crop0{int(crop_prob*10)}_flip0{int(h_flip_prob*10)}_rotate0{int(rotate_proba*10)}_{information_training}"

      if same >=1 :
        metrics_list = [epoch_avg_losses,epoch_avg_loss_classifier,epoch_avg_loss_box_reg,epoch_avg_loss_objectness,epoch_avg_loss_rpn_box_reg,epoch_avg_loss_keypoints,epoch_precision,epoch_recall,epoch_f1_score,epoch_test_loss]
        torch.save(best_model_state, './models/'+ name_model +'.pth')
        write_results(name_model,metrics_list,start_epoch)
        break

      if (epoch+1+start_epoch) % 5 == 0:
        metrics_list = [epoch_avg_losses,epoch_avg_loss_classifier,epoch_avg_loss_box_reg,epoch_avg_loss_objectness,epoch_avg_loss_rpn_box_reg,epoch_avg_loss_keypoints,epoch_precision,epoch_recall,epoch_f1_score,epoch_test_loss]
        torch.save(best_model_state, './models/'+ name_model +'.pth')
        model.load_state_dict(best_model_state)
        write_results(name_model,metrics_list,start_epoch)

      if avg_test_loss > previous_test_loss:
        bad_test_loss += 1
      previous_test_loss = avg_test_loss


  print(f"\n Total time: {(time.time() - start_tot)/60} minutes, Best Epoch is {best_epoch} with an f1_score of {best_metrics:.4f}")
  if best_model_state:
      metrics_list = [epoch_avg_losses,epoch_avg_loss_classifier,epoch_avg_loss_box_reg,epoch_avg_loss_objectness,epoch_avg_loss_rpn_box_reg,epoch_avg_loss_keypoints,epoch_precision,epoch_recall,epoch_f1_score,epoch_test_loss]
      torch.save(best_model_state, './models/'+ name_model +'.pth')
      model.load_state_dict(best_model_state)
      write_results(name_model,metrics_list,start_epoch)
      print(f"Name of the best model: model_{type(optimizer).__name__}_{epoch+1+start_epoch}ep_{batch_size}batch_trainval_blur0{int(blur_prob*10)}_crop0{int(crop_prob*10)}_flip0{int(h_flip_prob*10)}_rotate0{int(rotate_proba*10)}_{information_training}")

  return model, metrics_list