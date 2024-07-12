# utils.py
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F
import gc
import psutil
import copy
import xml.etree.ElementTree as ET
import numpy as np
from xml.dom import minidom
from pathlib import Path
import gdown



from modules.OCR import text_prediction, filter_text, mapping_text, rescale
from modules.utils import class_dict, arrow_dict, object_dict
from modules.toXML import calculate_pool_bounds, add_diagram_elements, create_bpmn_object, create_flow_element
from modules.eval import full_prediction
from modules.train import get_faster_rcnn_model, get_arrow_model
import torch
from pathlib import Path
import gdown
import os


def load_models():
    model_object = get_faster_rcnn_model(len(object_dict))
    model_arrow = get_arrow_model(len(arrow_dict), 2)

    # Paths to the model files in the Docker image
    model_object_path = 'models/model_object.pth'
    model_arrow_path = 'models/model_arrow.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_arrow.load_state_dict(torch.load(model_arrow_path, map_location=device))
    model_object.load_state_dict(torch.load(model_object_path, map_location=device))

    return model_object, model_arrow



def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

def clear_memory():
    gc.collect()

def modif_box_pos(pred, size):
    modified_pred = copy.deepcopy(pred)  # Make a deep copy of the prediction
    for i, (x1, y1, x2, y2) in enumerate(modified_pred['boxes']):
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        label = class_dict[modified_pred['labels'][i]]
        if label in size:
            modified_pred['boxes'][i] = [center[0] - size[label][0] / 2, center[1] - size[label][1] / 2, center[0] + size[label][0] / 2, center[1] + size[label][1] / 2]
    return modified_pred['boxes']

def create_XML(full_pred, text_mapping, scale=1, size_scale=1):
    namespaces = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'di': 'http://www.omg.org/spec/DD/20100524/DI',
        'dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }

    size_elements = {
        'event': (size_scale*43.2, size_scale*43.2),
        'task': (size_scale*120, size_scale*96),
        'message': (size_scale*43.2, size_scale*43.2),
        'messageEvent': (size_scale*43.2, size_scale*43.2),
        'exclusiveGateway': (size_scale*60, size_scale*60),
        'parallelGateway': (size_scale*60, size_scale*60),
        'dataObject': (size_scale*48, size_scale*72),
        'dataStore': (size_scale*72, size_scale*72),
        'subProcess': (size_scale*144, size_scale*108),
        'eventBasedGateway': (size_scale*60, size_scale*60),
        'timerEvent': (size_scale*48, size_scale*48),
    }

    definitions = ET.Element('bpmn:definitions', {
        'xmlns:xsi': namespaces['xsi'],
        'xmlns:bpmn': namespaces['bpmn'],
        'xmlns:bpmndi': namespaces['bpmndi'],
        'xmlns:di': namespaces['di'],
        'xmlns:dc': namespaces['dc'],
        'targetNamespace': "http://example.bpmn.com",
        'id': "simpleExample"
    })

    old_boxes = copy.deepcopy(full_pred)
    full_pred['boxes'] = modif_box_pos(full_pred, size_elements)

    collaboration = ET.SubElement(definitions, 'bpmn:collaboration', id='collaboration_1')

    process = []
    for idx in range(len(full_pred['pool_dict'].items())):
        process_id = f'process_{idx+1}'
        process.append(ET.SubElement(definitions, 'bpmn:process', id=process_id, isExecutable='false', name=text_mapping[full_pred['BPMN_id'][list(full_pred['pool_dict'].keys())[idx]]]))

    bpmndi = ET.SubElement(definitions, 'bpmndi:BPMNDiagram', id='BPMNDiagram_1')
    bpmnplane = ET.SubElement(bpmndi, 'bpmndi:BPMNPlane', id='BPMNPlane_1', bpmnElement='collaboration_1')

    full_pred['boxes'] = rescale(scale, full_pred['boxes'])

    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        pool_id = f'participant_{idx+1}'
        pool = ET.SubElement(collaboration, 'bpmn:participant', id=pool_id, processRef=f'process_{idx+1}', name=text_mapping[full_pred['BPMN_id'][list(full_pred['pool_dict'].keys())[idx]]])
        
        if len(keep_elements) == 0:
            min_x, min_y, max_x, max_y = full_pred['boxes'][pool_index]
            pool_width = max_x - min_x
            pool_height = max_y - min_y
        else:
            min_x, min_y, max_x, max_y = calculate_pool_bounds(full_pred, keep_elements, size_elements)
            pool_width = max_x - min_x + 100
            pool_height = max_y - min_y + 100
        
        add_diagram_elements(bpmnplane, pool_id, min_x - 50, min_y - 50, pool_width, pool_height)

    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        create_bpmn_object(process[idx], bpmnplane, text_mapping, definitions, size_elements, full_pred, keep_elements)

    message_flows = [i for i, label in enumerate(full_pred['labels']) if class_dict[label] == 'messageFlow']
    for idx in message_flows:
        create_flow_element(bpmnplane, text_mapping, idx, size_elements, full_pred, collaboration, message=True)

    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        for i in keep_elements:
            if full_pred['labels'][i] == list(class_dict.values()).index('sequenceFlow'):
                create_flow_element(bpmnplane, text_mapping, i, size_elements, full_pred, process[idx], message=False)
    
    tree = ET.ElementTree(definitions)
    rough_string = ET.tostring(definitions, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    full_pred['boxes'] = rescale(1/scale, full_pred['boxes'])
    full_pred['boxes'] = old_boxes

    return pretty_xml_as_string

def prepare_image(image, pad=True, new_size=(1333, 1333)):
    original_size = image.size
    scale = min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    new_scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    image = F.resize(image, (new_scaled_size[1], new_scaled_size[0]))

    if pad:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.0)
        padding = [0, 0, new_size[0] - new_scaled_size[0], new_size[1] - new_scaled_size[1]]
        image = F.pad(image, padding, fill=200, padding_mode='edge')

    return new_scaled_size, image

def perform_inference(model_object, model_arrow, image, score_threshold):
    _, uploaded_image = prepare_image(image, pad=False)
              
    img_tensor = F.to_tensor(prepare_image(image.convert('RGB'))[1])

    _, prediction = full_prediction(model_object, model_arrow, img_tensor, score_threshold=score_threshold, iou_threshold=0.5, distance_treshold=30)

    ocr_results = text_prediction(uploaded_image)
    text_pred = filter_text(ocr_results, threshold=0.6)
    text_mapping = mapping_text(prediction, text_pred, print_sentences=False, percentage_thresh=0.5)
                
    gc.collect()
    return prediction, text_mapping
