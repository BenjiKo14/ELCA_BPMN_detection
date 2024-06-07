from utils import draw_annotations, create_loader, class_dict, resize_boxes, resize_keypoints, find_other_keypoint
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from OCR import group_texts




def draw_stream(image, 
                prediction=None, 
                text_predictions=None, 
                class_dict=class_dict, 
                draw_keypoints=False, 
                draw_boxes=False, 
                draw_text=False,
                draw_links=False,
                draw_twins=False,
                draw_grouped_text=False,
                write_class=False,
                write_score=False, 
                write_text=False,
                score_threshold=0.4, 
                write_idx=False,
                keypoints_correction=False,
                new_size=(1333, 1333),
                only_print=None,
                axis=False,
                return_image=False,
                resize=False):
    """
    Draws annotations on images including bounding boxes, keypoints, links, and text.
    
    Parameters:
    - image (np.array): The image on which annotations will be drawn.
    - target (dict): Ground truth data containing boxes, labels, etc.
    - prediction (dict): Prediction data from a model.
    - full_prediction (dict): Additional detailed prediction data, potentially including relationships.
    - text_predictions (tuple): OCR text predictions containing bounding boxes and texts.
    - class_dict (dict): Mapping from class IDs to class names.
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

    
    image_copy = image.copy()
    scale = max(image.shape[0], image.shape[1]) / 1000

    original_size = (image.shape[0], image.shape[1])
    # Calculate scale to fit the new size while maintaining aspect ratio
    scale_ = min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    new_scaled_size = (int(original_size[0] * scale_), int(original_size[1] * scale_))

    for i in range(len(prediction['boxes'])):
        box = prediction['boxes'][i]
        x1, y1, x2, y2 = box
        if resize:
            x1, y1, x2, y2 = resize_boxes(np.array([box]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
        score = prediction['scores'][i]
        if score < score_threshold:
            continue
        if draw_boxes:
            if only_print is not None and only_print != 'all':
                if prediction['labels'][i] != list(class_dict.values()).index(only_print):
                    continue
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), int(2*scale))
        if write_score:
            cv2.putText(image_copy, str(round(score, 2)), (int(x1), int(y1) + int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (100,100, 255), 2)
        if write_idx:
            cv2.putText(image_copy, str(i), (int(x1) + int(15*scale), int(y1) + int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 2*scale, (0,0, 0), 2)

        if write_class and 'labels' in prediction:
            class_id = prediction['labels'][i]
            cv2.putText(image_copy, class_dict[class_id], (int(x1), int(y1) - int(2*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (255, 100, 100), 2)


        # Draw keypoints if available
        if draw_keypoints and 'keypoints' in prediction:
            for i in range(len(prediction['keypoints'])):
                kp = prediction['keypoints'][i]
                for j in range(kp.shape[0]):
                    if prediction['labels'][i] != list(class_dict.values()).index('sequenceFlow') and prediction['labels'][i] != list(class_dict.values()).index('messageFlow') and prediction['labels'][i] != list(class_dict.values()).index('dataAssociation'):
                        continue
                    
                    score = prediction['scores'][i]
                    if score < score_threshold:
                        continue
                    x,y, v = np.array(kp[j])
                    x, y, v = resize_keypoints(np.array([kp[j]]), (new_scaled_size[1],new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
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
                x1, y1, x2, y2 = resize_boxes(np.array([[float(x1), float(y1), float(x2), float(y2)]]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
            if draw_text:
                cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), int(2*scale))
            if write_text:
                cv2.putText(image_copy, text, (int(x1 + int(2*scale)), int((y1+y2)/2) ), cv2.FONT_HERSHEY_SIMPLEX, scale/2, (0,0, 0), 2)
            
    
    '''Draws links between objects based on the full prediction data.'''
    #check if keypoints detected are the same
    if draw_twins and prediction is not None:
        # Pre-calculate indices for performance
        circle_color = (0, 255, 0)  # Green color for the circle
        circle_radius = int(10 * scale)  # Circle radius scaled by image scale

        for idx, (key1, key2) in enumerate(prediction['keypoints']):
            if prediction['labels'][idx] not in [list(class_dict.values()).index('sequenceFlow'),
                    list(class_dict.values()).index('messageFlow'),
                    list(class_dict.values()).index('dataAssociation')]:
                continue
            # Calculate the Euclidean distance between the two keypoints
            distance = np.linalg.norm(key1[:2] - key2[:2])
            if distance < 10:
                x_new,y_new, x,y = find_other_keypoint(idx,prediction)
                cv2.circle(image_copy, (int(x), int(y)), circle_radius, circle_color, -1)
                cv2.circle(image_copy, (int(x_new), int(y_new)), circle_radius, (0,0,0), -1)

    # Draw links between objects
    if draw_links==True and prediction is not None:
        for arrow in prediction['links']:
            for i, (start_idx, end_idx) in enumerate(prediction['links'][arrow]):
                if start_idx is None or end_idx is None:
                    continue
                start_box = prediction['boxes'][start_idx]
                start_box = resize_boxes(np.array([start_box]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
                end_box = prediction['boxes'][end_idx]
                end_box = resize_boxes(np.array([end_box]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
                current_box = prediction['boxes'][i]
                current_box = resize_boxes(np.array([current_box]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
                # Calculate the center of each bounding box
                start_center = ((start_box[0] + start_box[2]) // 2, (start_box[1] + start_box[3]) // 2)
                end_center = ((end_box[0] + end_box[2]) // 2, (end_box[1] + end_box[3]) // 2)
                current_center = ((current_box[0] + current_box[2]) // 2, (current_box[1] + current_box[3]) // 2)
                # Draw a line between the centers of the connected objects
                cv2.line(image_copy, (int(start_center[0]), int(start_center[1])), (int(current_center[0]), int(current_center[1])), (0, 0, 255), int(2*scale))
                cv2.line(image_copy, (int(current_center[0]), int(current_center[1])), (int(end_center[0]), int(end_center[1])), (255, 0, 0), int(2*scale))

    if draw_grouped_text and prediction is not None:
        task_boxes = task_boxes = [box for i, box in enumerate(prediction['boxes']) if prediction['labels'][i] == list(class_dict.values()).index('task')]
        grouped_sentences, sentence_bounding_boxes, info_texts, info_boxes = group_texts(task_boxes, text_predictions[0], text_predictions[1], percentage_thresh=1)
        for i in range(len(info_boxes)):
            x1, y1, x2, y2 = info_boxes[i]
            x1, y1, x2, y2 = resize_boxes(np.array([[float(x1), float(y1), float(x2), float(y2)]]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), int(2*scale))
        for i in range(len(sentence_bounding_boxes)):
            x1,y1,x2,y2 = sentence_bounding_boxes[i]
            x1, y1, x2, y2 = resize_boxes(np.array([[float(x1), float(y1), float(x2), float(y2)]]), (new_scaled_size[1], new_scaled_size[0]), (image_copy.shape[1],image_copy.shape[0]))[0]
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), int(2*scale))
    
    if return_image:
        return image_copy
    else:
        # Display the image
        plt.figure(figsize=(12, 12))
        plt.imshow(image_copy)
        if axis==False:
            plt.axis('off')
        plt.show()