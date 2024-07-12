import numpy as np
import torch
from modules.utils import class_dict, object_dict, arrow_dict, find_closest_object, find_other_keypoint, filter_overlap_boxes, iou
from tqdm import tqdm
from modules.toXML import get_size_elements, calculate_pool_bounds, create_BPMN_id
from modules.utils import is_vertical
import streamlit as st


def non_maximum_suppression(boxes, scores, labels=None, iou_threshold=0.5):
    exception = ['pool', 'lane']

    idxs = np.argsort(scores)  # Sort the boxes according to their scores in ascending order
    selected_boxes = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]

        # Skip if the label is a lane
        if labels is not None and (class_dict[labels[i]] in exception):
            selected_boxes.append(i)
            idxs = np.delete(idxs, last)
            continue

        selected_boxes.append(i)

        # Find the intersection of the box with the rest
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    # Return only the boxes that were selected
    return selected_boxes


def keypoint_correction(keypoints, boxes, labels, model_dict=arrow_dict, distance_treshold=15):
    for idx, (key1, key2) in enumerate(keypoints):
            if labels[idx] not in [list(model_dict.values()).index('sequenceFlow'),
                        list(model_dict.values()).index('messageFlow'),
                        list(model_dict.values()).index('dataAssociation')]:
                continue
            # Calculate the Euclidean distance between the two keypoints
            distance = np.linalg.norm(key1[:2] - key2[:2])
            if distance < distance_treshold:
                print('Key modified for index:', idx)
                x_new,y_new, x,y = find_other_keypoint(idx, keypoints, boxes)
                keypoints[idx][0][:2] = [x_new,y_new]
                keypoints[idx][1][:2] = [x,y]

    return keypoints


def object_prediction(model, image, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        predictions = model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        idx = np.where(scores > score_threshold)[0]
        boxes = boxes[idx]
        scores = scores[idx]
        labels = labels[idx]

        selected_boxes = non_maximum_suppression(boxes, scores, labels=labels, iou_threshold=iou_threshold)

        #find orientation of the task by checking the size of all the boxes and delete the one that are not in the same orientation
        vertical = 0
        for i in range(len(labels)):
            if labels[i] != list(object_dict.values()).index('task'):
                continue
            if is_vertical(boxes[i]):
                vertical += 1
        horizontal = len(labels) - vertical
        for i in range(len(labels)):
            if labels[i] != list(object_dict.values()).index('task'):
                continue

            if vertical < horizontal:
                if is_vertical(boxes[i]):
                    #find the element in the list and remove it
                    if i in selected_boxes:
                        selected_boxes.remove(i)
            elif vertical > horizontal:
                if is_vertical(boxes[i]) == False:
                    #find the element in the list and remove it
                    if i in selected_boxes:
                        selected_boxes.remove(i)
            else:
                pass

        boxes = boxes[selected_boxes]
        scores = scores[selected_boxes]
        labels = labels[selected_boxes]

        #modify the label of the sub-process to task
        for i in range(len(labels)):
            if labels[i] == list(object_dict.values()).index('subProcess'):
                labels[i] = list(object_dict.values()).index('task')

        prediction = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
        }

    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    return image, prediction


def arrow_prediction(model, image, score_threshold=0.5, iou_threshold=0.5, distance_treshold=15):
    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        predictions = model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy() + (len(object_dict) - 1)
        scores = predictions[0]['scores'].cpu().numpy()
        keypoints = predictions[0]['keypoints'].cpu().numpy()

        idx = np.where(scores > score_threshold)[0]
        boxes = boxes[idx]
        scores = scores[idx]
        labels = labels[idx]
        keypoints = keypoints[idx]

        selected_boxes = non_maximum_suppression(boxes, scores, iou_threshold=iou_threshold)
        boxes = boxes[selected_boxes]
        scores = scores[selected_boxes]
        labels = labels[selected_boxes]
        keypoints = keypoints[selected_boxes]

        keypoints = keypoint_correction(keypoints, boxes, labels, class_dict, distance_treshold=distance_treshold)

        prediction = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'keypoints': keypoints,
        }

    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    return image, prediction

def mix_predictions(objects_pred, arrow_pred):
    # Initialize the list of lists for keypoints
    object_keypoints = []

    # Number of boxes
    num_boxes = len(objects_pred['boxes'])

    # Iterate over the number of boxes
    for _ in range(num_boxes):
        # Each box has 2 keypoints, both initialized to [0, 0, 0]
        keypoints = [[0, 0, 0], [0, 0, 0]]
        object_keypoints.append(keypoints)

    #concatenate the two predictions
    if len(arrow_pred['boxes']) == 0:
        return objects_pred['boxes'], objects_pred['labels'], objects_pred['scores'], object_keypoints
    
    boxes = np.concatenate((objects_pred['boxes'], arrow_pred['boxes']))
    labels = np.concatenate((objects_pred['labels'], arrow_pred['labels']))
    scores = np.concatenate((objects_pred['scores'], arrow_pred['scores']))
    keypoints = np.concatenate((object_keypoints, arrow_pred['keypoints']))

    return boxes, labels, scores, keypoints


def regroup_elements_by_pool(boxes, labels, scores, keypoints, class_dict, iou_threshold=0.3):
    pool_dict = {}

    # Filter out pools with IoU greater than the threshold
    to_delete = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if labels[i] == labels[j] and labels[i] == list(class_dict.values()).index('pool'):
                if iou(np.array(boxes[i]), np.array(boxes[j])) > iou_threshold:
                    to_delete.append(j)

    boxes = np.delete(boxes, to_delete, axis=0)
    labels = np.delete(labels, to_delete)
    scores = np.delete(scores, to_delete)
    keypoints = np.delete(keypoints, to_delete, axis=0)

    pool_indices = [i for i, label in enumerate(labels) if class_dict[label.item()] == 'pool']
    pool_boxes = [boxes[i] for i in pool_indices]

    if pool_indices:
        for pool_index in pool_indices:
            pool_dict[pool_index] = []

        elements_not_in_pool = []

        for i, box in enumerate(boxes):
            assigned_to_pool = False
            if i in pool_indices or class_dict[labels[i]] in ['messageFlow', 'pool']:
                continue
            for j, pool_box in enumerate(pool_boxes):
                if (box[0] >= pool_box[0] and box[1] >= pool_box[1] and 
                    box[2] <= pool_box[2] and box[3] <= pool_box[3]):
                    pool_index = pool_indices[j]
                    pool_dict[pool_index].append(i)
                    assigned_to_pool = True
                    break
            if not assigned_to_pool:
                if class_dict[labels[i]] not in ['messageFlow', 'lane', 'pool']:
                    elements_not_in_pool.append(i)

        if len(elements_not_in_pool) > 1:
            new_elements_not_in_pool = [i for i in elements_not_in_pool if class_dict[labels[i]] not in ['messageFlow', 'lane', 'pool']]
            
            # Indices of relevant classes
            sequence_flow_index = list(class_dict.values()).index('sequenceFlow')
            message_flow_index = list(class_dict.values()).index('messageFlow')
            data_association_index = list(class_dict.values()).index('dataAssociation')

            if all(labels[i] in {sequence_flow_index, message_flow_index, data_association_index} for i in new_elements_not_in_pool):
                print('The new pool contains only sequenceFlow, messageFlow, or dataAssociation') 

            elif len(new_elements_not_in_pool) > 1:
                new_pool_index = len(labels)
                box = calculate_pool_bounds(boxes, labels, new_elements_not_in_pool, None)
                boxes = np.append(boxes, [box], axis=0)
                labels = np.append(labels, list(class_dict.values()).index('pool'))
                scores = np.append(scores, 1.0)
                keypoints = np.append(keypoints, np.zeros((1, 2, 3)), axis=0)
                pool_dict[new_pool_index] = new_elements_not_in_pool
                print(f"Created a new pool index {new_pool_index} with elements: {new_elements_not_in_pool}")
       
    non_empty_pools = {k: v for k, v in pool_dict.items() if v}
    empty_pools = {k: v for k, v in pool_dict.items() if not v}
    pool_dict = {**non_empty_pools, **empty_pools}

    return pool_dict, boxes, labels, scores, keypoints



def create_links(keypoints, boxes, labels, class_dict):
    best_points = []
    links = []
    for i in range(len(labels)):
        if labels[i]==list(class_dict.values()).index('sequenceFlow') or labels[i]==list(class_dict.values()).index('messageFlow'):
            closest1, point_start = find_closest_object(keypoints[i][0], boxes, labels)
            closest2, point_end = find_closest_object(keypoints[i][1], boxes, labels)
            
            if closest1 is not None and closest2 is not None:
                best_points.append([point_start, point_end])
                links.append([closest1, closest2])
        else:
            best_points.append([None,None])
            links.append([None,None])

    for i in range(len(labels)):
        if labels[i]==list(class_dict.values()).index('dataAssociation'):
            closest1, point_start = find_closest_object(keypoints[i][0], boxes, labels)
            closest2, point_end = find_closest_object(keypoints[i][1], boxes, labels)
            if closest1 is not None and closest2 is not None:
                best_points[i] = ([point_start, point_end])
                links[i] = ([closest1, closest2])

    return links, best_points

def correction_labels(boxes, labels, class_dict, pool_dict, flow_links):
    sequence_flow_index = list(class_dict.values()).index('sequenceFlow')
    message_flow_index = list(class_dict.values()).index('messageFlow')
    data_association_index = list(class_dict.values()).index('dataAssociation')
    data_object_index = list(class_dict.values()).index('dataObject')
    data_store_index = list(class_dict.values()).index('dataStore')
    message_event_index = list(class_dict.values()).index('messageEvent')
    senquence_flow_indexx = list(class_dict.values()).index('sequenceFlow')

    for pool_index, elements in pool_dict.items():
        print(f"Pool {pool_index} contains elements: {elements}")
        
        # Check if the label sequenceFlow or messageFlow is good
        for i, (id1, id2) in enumerate(flow_links):
            if labels[i] in {sequence_flow_index, message_flow_index}:
                if id1 is not None and id2 is not None:
                    # Check if each link is in the same pool
                    if id1 in elements and id2 in elements:
                        # Check if the link is between a dataObject or a dataStore
                        if labels[id1] in {data_object_index, data_store_index} or labels[id2] in {data_object_index, data_store_index}:
                            print('Change the link from sequenceFlow/messageFlow to dataAssociation')
                            labels[i] = data_association_index
                        else:
                            continue
                    elif id1 not in elements and id2 not in elements:
                        continue
                    else:
                        print('Change the link from sequenceFlow to messageFlow')
                        labels[i] = message_flow_index

    # Check if dataAssociation is connected to a dataObject
    for i, (id1, id2) in enumerate(flow_links):
        if labels[i] == data_association_index:
            if id1 is not None and id2 is not None:
                label1 = labels[id1]
                label2 = labels[id2]
                if data_object_index in {label1, label2} or data_store_index in {label1, label2}:
                    continue
                elif message_event_index in {label1, label2}: 
                    print('Change the link from dataAssociation to messageFlow')
                    labels[i] = message_flow_index
                else:
                    print('Change the link from dataAssociation to sequenceFlow')
                    labels[i] = senquence_flow_indexx

    return labels, flow_links



def last_correction(boxes, labels, scores, keypoints, bpmn_id, links, best_points, pool_dict, limit_area=10000):

    #delete pool that are have only messageFlow on it
    delete_pool = []
    for pool_index, elements in pool_dict.items():
        #find the position of the pool_index in the bpmn_id
        if pool_index in bpmn_id:
            position = bpmn_id.index(pool_index)
        else:
            continue 
        if all([labels[i] in [list(class_dict.values()).index('messageFlow'),
                              list(class_dict.values()).index('sequenceFlow'),
                              list(class_dict.values()).index('dataAssociation'),
                              list(class_dict.values()).index('lane')] for i in elements]):
            if len(elements) > 0:
                delete_pool.append(position)
                print(f"Pool {pool_index} contains only arrow elements, deleting it")   

        #calcul the area of the pool$
        if position < len(boxes):
            pool = boxes[position]
            area = (pool[2] - pool[0]) * (pool[3] - pool[1])
            if len(pool_dict)>1 and area < limit_area:
                delete_pool.append(position)
                print(f"Pool {pool_index} is too small, deleting it")     

            if is_vertical(boxes[position]):
                delete_pool.append(position)
                print(f"Pool {position} is vertical, deleting it")


    delete_elements = []
    # Check if there is an arrow that has the same links
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == list(class_dict.values()).index('sequenceFlow') and labels[j] == list(class_dict.values()).index('sequenceFlow'):
                if links[i] == links[j]:
                    print(f'element {i} and {j} have the same links')
                    if scores[i] > scores[j]:
                        print('delete element', j)
                        delete_elements.append(j)
                    else:
                        print('delete element', i)
                        delete_elements.append(i)

    #concatenate the delete_elements and the delete_pool
    delete_elements = delete_elements + delete_pool
    #delete double value in delete_elements
    delete_elements = list(set(delete_elements))

    boxes = np.delete(boxes, delete_elements, axis=0)
    labels = np.delete(labels, delete_elements)
    scores = np.delete(scores, delete_elements)
    keypoints = np.delete(keypoints, delete_elements, axis=0)
    bpmn_id = [point for i, point in enumerate(bpmn_id) if i not in delete_elements]
    links = np.delete(links, delete_elements, axis=0)
    best_points = [point for i, point in enumerate(best_points) if i not in delete_elements]

    #also delete the element in the pool_dict
    for pool_index, elements in pool_dict.items():
        pool_dict[pool_index] = [i for i in elements if i not in delete_elements]

    return boxes, labels, scores, keypoints, bpmn_id, links, best_points, pool_dict

def give_link_to_element(links, labels):
    #give a link to event to allow the creation of the BPMN id with start, indermediate and end event
        for i in range(len(links)):
            if labels[i] == list(class_dict.values()).index('sequenceFlow'):
                id1, id2 = links[i]
                if (id1 and id2) is not None:
                        links[id1][1] = i
                        links[id2][0] = i
        return links


def generate_data(image, boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict):
            idx = []
            for i in range(len(labels)):
                idx.append(i) 
              

            data = {
                'image': image,
                'idx': idx,
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'keypoints': keypoints,
                'links': flow_links,
                'best_points': best_points,
                'pool_dict': pool_dict,
                'BPMN_id': bpmn_id,
            }


            return data

def develop_prediction(boxes, labels, scores, keypoints, class_dict, correction=True):

    pool_dict, boxes, labels, scores, keypoints = regroup_elements_by_pool(boxes, labels, scores, keypoints, class_dict)

    bpmn_id, pool_dict = create_BPMN_id(labels,pool_dict)

    # Create links between elements
    flow_links, best_points = create_links(keypoints, boxes, labels, class_dict)
    
    #Correct the labels of some sequenceflow that cross multiple pool
    if correction:
        labels, flow_links = correction_labels(boxes, labels, class_dict, pool_dict, flow_links)
    
    #give a link to event to allow the creation of the BPMN id with start, indermediate and end event
    flow_links = give_link_to_element(flow_links, labels)
    
    boxes,labels,scores,keypoints,bpmn_id, flow_links,best_points,pool_dict = last_correction(boxes,labels,scores,keypoints,bpmn_id,flow_links,best_points, pool_dict)  

    return boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict 

    

def full_prediction(model_object, model_arrow, image, score_threshold=0.5, iou_threshold=0.5, resize=True, distance_treshold=15):
    model_object.eval()  # Set the model to evaluation mode
    model_arrow.eval()  # Set the model to evaluation mode

    # Load an image
    with torch.no_grad():  # Disable gradient calculation for inference
        _, objects_pred = object_prediction(model_object, image, score_threshold=score_threshold, iou_threshold=0.1)
        _, arrow_pred = arrow_prediction(model_arrow, image, score_threshold=score_threshold, iou_threshold=iou_threshold, distance_treshold=distance_treshold)

        st.session_state.arrow_pred = arrow_pred
        
        boxes, labels, scores, keypoints = mix_predictions(objects_pred, arrow_pred)
        
        boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict = develop_prediction(boxes, labels, scores, keypoints, class_dict)
        
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
        data = generate_data(image, boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict)

        return image, data

def evaluate_model_by_class(pred_boxes, true_boxes, pred_labels, true_labels, model_dict, iou_threshold=0.5):
    # Initialize dictionaries to hold per-class counts
    class_tp = {cls: 0 for cls in model_dict.values()}
    class_fp = {cls: 0 for cls in model_dict.values()}
    class_fn = {cls: 0 for cls in model_dict.values()}

    # Track which true boxes have been matched
    matched = [False] * len(true_boxes)

    # Check each prediction against true boxes
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        match_found = False
        for idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if not matched[idx] and pred_label == true_label:
                if iou(np.array(pred_box), np.array(true_box)) >= iou_threshold:
                    class_tp[model_dict[pred_label]] += 1
                    matched[idx] = True
                    match_found = True
                    break
        if not match_found:
            class_fp[model_dict[pred_label]] += 1

    # Count false negatives
    for idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
        if not matched[idx]:
            class_fn[model_dict[true_label]] += 1

    # Calculate precision, recall, and F1-score per class
    class_precision = {}
    class_recall = {}
    class_f1_score = {}

    for cls in model_dict.values():
        precision = class_tp[cls] / (class_tp[cls] + class_fp[cls]) if class_tp[cls] + class_fp[cls] > 0 else 0
        recall = class_tp[cls] / (class_tp[cls] + class_fn[cls]) if class_tp[cls] + class_fn[cls] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        class_precision[cls] = precision
        class_recall[cls] = recall
        class_f1_score[cls] = f1_score

    return class_precision, class_recall, class_f1_score


def keypoints_mesure(pred_boxes, pred_box, true_boxes, true_box, pred_keypoints, true_keypoints, distance_threshold=5):
    result = 0
    reverted = False
    #find the position of keypoints in the list
    idx = np.where(pred_boxes == pred_box)[0][0]
    idx2 = np.where(true_boxes == true_box)[0][0]

    keypoint1_pred = pred_keypoints[idx][0]
    keypoint1_true = true_keypoints[idx2][0]
    keypoint2_pred = pred_keypoints[idx][1]
    keypoint2_true = true_keypoints[idx2][1]

    distance1 = np.linalg.norm(keypoint1_pred[:2] - keypoint1_true[:2])
    distance2 = np.linalg.norm(keypoint2_pred[:2] - keypoint2_true[:2])
    distance3 = np.linalg.norm(keypoint1_pred[:2] - keypoint2_true[:2])
    distance4 = np.linalg.norm(keypoint2_pred[:2] - keypoint1_true[:2])

    if distance1 < distance_threshold:
        result += 1
    if distance2 < distance_threshold:
        result += 1
    if distance3 < distance_threshold or distance4 < distance_threshold:
        reverted = True

    return result, reverted

def evaluate_single_image(pred_boxes, true_boxes, pred_labels, true_labels, pred_keypoints, true_keypoints, iou_threshold=0.5, distance_threshold=5):
    tp, fp, fn = 0, 0, 0
    key_t, key_f = 0, 0
    labels_t, labels_f = 0, 0
    reverted_tot = 0

    matched_true_boxes = set()
    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        match_found = False
        for true_idx, true_box in enumerate(true_boxes):
            if true_idx in matched_true_boxes:
                continue
            iou_val = iou(pred_box, true_box)
            if iou_val >= iou_threshold:
                if true_keypoints is not None and pred_keypoints is not None:
                    key_result, reverted = keypoints_mesure(pred_boxes, pred_box, true_boxes, true_box, pred_keypoints, true_keypoints, distance_threshold)
                    key_t += key_result
                    key_f += 2 - key_result
                    if reverted:
                        reverted_tot += 1
            
                match_found = True
                matched_true_boxes.add(true_idx)
                if pred_label == true_labels[true_idx]:
                    labels_t += 1
                else:
                    labels_f += 1
                tp += 1
                break
        if not match_found:
            fp += 1

    fn = len(true_boxes) - tp

    return tp, fp, fn, labels_t, labels_f, key_t, key_f, reverted_tot


def pred_4_evaluation(model, loader, score_threshold=0.5, iou_threshold=0.5, distance_threshold=5, key_correction=True, model_type='object'):
    model.eval()
    tp, fp, fn = 0, 0, 0
    labels_t, labels_f = 0, 0
    key_t, key_f = 0, 0
    reverted = 0

    with torch.no_grad():
        for images, targets_im in tqdm(loader, desc="Testing... "):  # Wrap the loader with tqdm
            devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            images = [image.to(devices) for image in images]
            targets = [{k: v.clone().detach().to(devices) for k, v in t.items()} for t in targets_im]

            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                if 'keypoints' in target:
                    true_keypoints = target['keypoints'].cpu().numpy()

                pred_boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()
                if 'keypoints' in prediction:
                    pred_keypoints = prediction['keypoints'].cpu().numpy()

                selected_boxes = non_maximum_suppression(pred_boxes, scores, iou_threshold=iou_threshold)
                pred_boxes = pred_boxes[selected_boxes]
                scores = scores[selected_boxes]
                pred_labels = pred_labels[selected_boxes]
                if 'keypoints' in prediction:
                    pred_keypoints = pred_keypoints[selected_boxes]

                filtered_boxes = []
                filtered_labels = []
                filtered_keypoints = []
                if 'keypoints' not in prediction:
                    #create a list of zeros of length equal to the number of boxes
                    pred_keypoints = [np.zeros((2, 3)) for _ in range(len(pred_boxes))]

                for box, score, label, keypoints in zip(pred_boxes, scores, pred_labels, pred_keypoints):
                    if score >= score_threshold:
                        filtered_boxes.append(box)
                        filtered_labels.append(label)
                        if 'keypoints' in prediction:
                            filtered_keypoints.append(keypoints)

                if key_correction and ('keypoints' in prediction):
                    filtered_keypoints = keypoint_correction(filtered_keypoints, filtered_boxes, filtered_labels)

                if 'keypoints' not in target:
                    filtered_keypoints = None
                    true_keypoints = None
                tp_img, fp_img, fn_img, labels_t_img, labels_f_img, key_t_img, key_f_img, reverted_img = evaluate_single_image(
                    filtered_boxes, true_boxes, filtered_labels, true_labels, filtered_keypoints, true_keypoints, iou_threshold, distance_threshold)

                tp += tp_img
                fp += fp_img
                fn += fn_img
                labels_t += labels_t_img
                labels_f += labels_f_img
                key_t += key_t_img
                key_f += key_f_img
                reverted += reverted_img

    return tp, fp, fn, labels_t, labels_f, key_t, key_f, reverted

def main_evaluation(model, test_loader, score_threshold=0.5, iou_threshold=0.5, distance_threshold=5, key_correction=True, model_type = 'object'):

    tp, fp, fn, labels_t, labels_f, key_t, key_f, reverted = pred_4_evaluation(model, test_loader, score_threshold, iou_threshold, distance_threshold, key_correction, model_type)

    labels_precision = labels_t / (labels_t + labels_f) if (labels_t + labels_f) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    if model_type == 'arrow':
        key_accuracy = key_t / (key_t + key_f) if (key_t + key_f) > 0 else 0
        reverted_accuracy = reverted / (key_t + key_f) if (key_t + key_f) > 0 else 0
    else:
        key_accuracy = 0
        reverted_accuracy = 0

    return labels_precision, precision, recall, f1_score, key_accuracy, reverted_accuracy



def evaluate_model_by_class_single_image(pred_boxes, true_boxes, pred_labels, true_labels, class_tp, class_fp, class_fn, model_dict, iou_threshold=0.5):
    matched_true_boxes = set()
    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        match_found = False
        for true_idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if true_idx in matched_true_boxes:
                continue
            if pred_label == true_label and iou(np.array(pred_box), np.array(true_box)) >= iou_threshold:
                class_tp[model_dict[pred_label]] += 1
                matched_true_boxes.add(true_idx)
                match_found = True
                break
        if not match_found:
            class_fp[model_dict[pred_label]] += 1

    for idx, true_label in enumerate(true_labels):
        if idx not in matched_true_boxes:
            class_fn[model_dict[true_label]] += 1

def pred_4_evaluation_per_class(model, loader, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        for images, targets_im in tqdm(loader, desc="Testing... "):
            devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            images = [image.to(devices) for image in images]
            targets = [{k: v.clone().detach().to(devices) for k, v in t.items()} for t in targets_im]

            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                pred_boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()

                idx = np.where(scores > score_threshold)[0]
                pred_boxes = pred_boxes[idx]
                scores = scores[idx]
                pred_labels = pred_labels[idx]

                selected_boxes = non_maximum_suppression(pred_boxes, scores, iou_threshold=iou_threshold)
                pred_boxes = pred_boxes[selected_boxes]
                scores = scores[selected_boxes]
                pred_labels = pred_labels[selected_boxes]

                yield pred_boxes, true_boxes, pred_labels, true_labels

def evaluate_model_by_class(model, test_loader, model_dict, score_threshold=0.5, iou_threshold=0.5):
    class_tp = {cls: 0 for cls in model_dict.values()}
    class_fp = {cls: 0 for cls in model_dict.values()}
    class_fn = {cls: 0 for cls in model_dict.values()}

    for pred_boxes, true_boxes, pred_labels, true_labels in pred_4_evaluation_per_class(model, test_loader, score_threshold, iou_threshold):
        evaluate_model_by_class_single_image(pred_boxes, true_boxes, pred_labels, true_labels, class_tp, class_fp, class_fn, model_dict, iou_threshold)

    class_precision = {}
    class_recall = {}
    class_f1_score = {}

    for cls in model_dict.values():
        precision = class_tp[cls] / (class_tp[cls] + class_fp[cls]) if class_tp[cls] + class_fp[cls] > 0 else 0
        recall = class_tp[cls] / (class_tp[cls] + class_fn[cls]) if class_tp[cls] + class_fn[cls] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        class_precision[cls] = precision
        class_recall[cls] = recall
        class_f1_score[cls] = f1_score

    return class_precision, class_recall, class_f1_score