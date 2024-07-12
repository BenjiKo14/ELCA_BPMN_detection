import xml.etree.ElementTree as ET
from modules.utils import class_dict, error, warning
import streamlit as st
from modules.utils import class_dict, rescale_boxes
import copy
from xml.dom import minidom
import numpy as np

def find_position(pool_index, BPMN_id):
    #find the position of the pool_index in the bpmn_id
    if pool_index in BPMN_id:
        position = BPMN_id.index(pool_index)
    else:
        position = None
        error(f"Problem with the pool index {pool_index} in the BPMN_id")

    return position

# Calculate the center of each bounding box and group them by pool
def calculate_centers_and_group_by_pool(pred, class_dict):
    pool_groups = {}
    for pool_index, element_indices in pred['pool_dict'].items():
        pool_groups[pool_index] = []
        for i in element_indices:
            if i >= len(pred['labels']):
                continue
            if class_dict[pred['labels'][i]] not in ['dataObject', 'dataStore']:
                x1, y1, x2, y2 = pred['boxes'][i]
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                pool_groups[pool_index].append((center, i))
    return pool_groups

# Group centers within a specified range
def group_centers(centers, axis, range_=50):
    groups = []
    while centers:
        center, idx = centers.pop(0)
        group = [(center, idx)]
        for other_center, other_idx in centers[:]:
            if abs(center[axis] - other_center[axis]) <= range_:
                group.append((other_center, other_idx))
                centers.remove((other_center, other_idx))
        groups.append(group)
    return groups

# Align the elements within each pool
def align_elements_within_pool(modified_pred, pool_groups, class_dict, size):
    for pool_index, centers in pool_groups.items():
        y_groups = group_centers(centers.copy(), axis=1)
        align_y_coordinates(modified_pred, y_groups, class_dict, size)
        
        centers = recalculate_centers(modified_pred, y_groups)
        x_groups = group_centers(centers.copy(), axis=0)
        align_x_coordinates(modified_pred, x_groups, class_dict, size)

# Align the y-coordinates of the centers of grouped bounding boxes
def align_y_coordinates(modified_pred, y_groups, class_dict, size):
    for group in y_groups:
        avg_y = sum([c[0][1] for c in group]) / len(group)
        for (center, idx) in group:
            label = class_dict[modified_pred['labels'][idx]]
            if label in size:
                new_center = (center[0], avg_y)
                modified_pred['boxes'][idx] = [
                    new_center[0] - size[label][0] / 2, 
                    new_center[1] - size[label][1] / 2, 
                    new_center[0] + size[label][0] / 2, 
                    new_center[1] + size[label][1] / 2
                ]

# Recalculate centers after alignment
def recalculate_centers(modified_pred, groups):
    centers = []
    for group in groups:
        for center, idx in group:
            x1, y1, x2, y2 = modified_pred['boxes'][idx]
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            centers.append((center, idx))
    return centers

# Align the x-coordinates of the centers of grouped bounding boxes
def align_x_coordinates(modified_pred, x_groups, class_dict, size):
    for group in x_groups:
        avg_x = sum([c[0][0] for c in group]) / len(group)
        for (center, idx) in group:
            label = class_dict[modified_pred['labels'][idx]]
            if label in size:
                new_center = (avg_x, center[1])
                modified_pred['boxes'][idx] = [
                    new_center[0] - size[label][0] / 2, 
                    modified_pred['boxes'][idx][1], 
                    new_center[0] + size[label][0] / 2, 
                    modified_pred['boxes'][idx][3]
                ]

# Expand the pool bounding boxes to fit the aligned elements
def expand_pool_bounding_boxes(modified_pred, pred, size_elements):
    for idx, (pool_index, keep_elements) in enumerate(modified_pred['pool_dict'].items()):
        if len(keep_elements) != 0:
            marge = 50
        else:
            marge = 0

        position = find_position(pool_index, modified_pred['BPMN_id'])

        if keep_elements == [] or position is None:
            min_x, min_y, max_x, max_y = modified_pred['boxes'][position]
        else:
            min_x, min_y, max_x, max_y = calculate_pool_bounds(modified_pred['boxes'], modified_pred['labels'], keep_elements, size_elements)

        pool_width = max_x - min_x
        pool_height = max_y - min_y
        if pool_width < 300 or pool_height < 30:
            error("The pool is maybe too small, please add more elements or increase the scale by zooming on the image.")
            continue

        modified_pred['boxes'][position] = [min_x - marge, min_y - marge // 2, min_x + pool_width + marge, min_y + pool_height + marge // 2]

# Adjust left and right boundaries of all pools
def adjust_pool_boundaries(modified_pred, pred):
    min_left, max_right = 0, 0
    for pool_index, element_indices in pred['pool_dict'].items():
        position = find_position(pool_index, modified_pred['BPMN_id'])
        if position >= len(modified_pred['boxes']):
            continue
        x1, y1, x2, y2 = modified_pred['boxes'][position]
        left = x1
        right = x2
        if left < min_left:
            min_left = left
        if right > max_right:
            max_right = right

    for pool_index, element_indices in pred['pool_dict'].items():
        position = find_position(pool_index, modified_pred['BPMN_id'])
        if position >= len(modified_pred['boxes']):
            continue
        x1, y1, x2, y2 = modified_pred['boxes'][position]
        if x1 > min_left:
            x1 = min_left
        if x2 < max_right:
            x2 = max_right
        modified_pred['boxes'][position] = [x1, y1, x2, y2]

# Main function to align boxes
def align_boxes(pred, size, class_dict):
    modified_pred = copy.deepcopy(pred)
    pool_groups = calculate_centers_and_group_by_pool(pred, class_dict)
    align_elements_within_pool(modified_pred, pool_groups, class_dict, size)
    
    if len(pred['pool_dict']) > 1:
        expand_pool_bounding_boxes(modified_pred, pred, size)
        adjust_pool_boundaries(modified_pred, pred)

    return modified_pred['boxes']


# Function to create a BPMN XML file from prediction results
def create_XML(full_pred, text_mapping, size_scale, scale):
    namespaces = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'di': 'http://www.omg.org/spec/DD/20100524/DI',
        'dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
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

    size_elements = get_size_elements(size_scale)

    #if there is no pool or lane, create a pool with all elements
    if len(full_pred['pool_dict'])==0 or (len(full_pred['pool_dict'])==1 and len(full_pred['pool_dict']['pool_1'])==len(full_pred['labels'])):
        full_pred, text_mapping = create_big_pool(full_pred, text_mapping)
    
    #modify the boxes positions
    old_boxes = copy.deepcopy(full_pred)

    # Create BPMN collaboration element
    collaboration = ET.SubElement(definitions, 'bpmn:collaboration', id='collaboration_1')

    # Create BPMN process elements
    process = []
    for idx in range (len(full_pred['pool_dict'].items())):
        process_id = f'process_{idx+1}'      
        process.append(ET.SubElement(definitions, 'bpmn:process', id=process_id, isExecutable='false'))

    bpmndi = ET.SubElement(definitions, 'bpmndi:BPMNDiagram', id='BPMNDiagram_1')
    bpmnplane = ET.SubElement(bpmndi, 'bpmndi:BPMNPlane', id='BPMNPlane_1', bpmnElement='collaboration_1')

    full_pred['boxes'] = rescale_boxes(scale, old_boxes['boxes'])
    full_pred['boxes'] = align_boxes(full_pred, size_elements, class_dict)
         

    # Add diagram elements for each pool
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        pool_id = f'participant_{idx+1}'
        pool = ET.SubElement(collaboration, 'bpmn:participant', id=pool_id, processRef=f'process_{idx+1}', name=text_mapping[pool_index])
        
        position = find_position(pool_index, full_pred['BPMN_id'])
        # Calculate the bounding box for the pool
        #if len(keep_elements) == 0:
        if position >= len(full_pred['boxes']):
            print("Problem with the index")
            continue
        min_x, min_y, max_x, max_y = full_pred['boxes'][position]
        pool_width = max_x - min_x
        pool_height = max_y - min_y
        
        add_diagram_elements(bpmnplane, pool_id, min_x, min_y, pool_width, pool_height)


    # Create BPMN elements for each pool
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        create_bpmn_object(process[idx], bpmnplane, text_mapping, definitions, size_elements, full_pred, keep_elements)

    # Create message flow elements
    message_flows = [i for i, label in enumerate(full_pred['labels']) if class_dict[label] == 'messageFlow']
    for idx in message_flows:
        create_flow_element(bpmnplane, text_mapping, idx, size_elements, full_pred, collaboration, message=True)

    # Create sequence flow elements
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        for i in keep_elements:
            if i >= len(full_pred['labels']):
                print("Problem with the index")
                continue
            if full_pred['labels'][i] == list(class_dict.values()).index('sequenceFlow'):
                create_flow_element(bpmnplane, text_mapping, i, size_elements, full_pred, process[idx], message=False)
    
    # Generate pretty XML string
    tree = ET.ElementTree(definitions)
    rough_string = ET.tostring(definitions, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    full_pred['boxes'] = rescale_boxes(1/scale, full_pred['boxes'])
    full_pred['boxes'] = old_boxes

    return pretty_xml_as_string

# Function that creates a single pool with all elements
def create_big_pool(full_pred, text_mapping):
    # If no pools or lanes are detected, create a single pool with all elements
    new_pool_index = 'pool_1'
    size_elements = get_size_elements(st.session_state.size_scale)
    elements_pool = list(range(len(full_pred['boxes'])))
    min_x, min_y, max_x, max_y = calculate_pool_bounds(full_pred['boxes'],full_pred['labels'], elements_pool, size_elements)
    box = [min_x, min_y, max_x, max_y]
    full_pred['boxes'] = np.append(full_pred['boxes'], [box], axis=0)
    full_pred['pool_dict'][new_pool_index] = elements_pool
    full_pred['BPMN_id'].append('pool_1')
    text_mapping['pool_1'] = 'Process'
    print(f"Created a big pool index {new_pool_index} with elements: {elements_pool}")
    return full_pred, text_mapping

# Function that gives the size of the elements
def get_size_elements(size_scale):
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
    return size_elements

def rescale(scale, boxes):
    for i in range(len(boxes)):
                boxes[i] = [boxes[i][0]*scale,
                            boxes[i][1]*scale,
                            boxes[i][2]*scale,
                            boxes[i][3]*scale]
    return boxes

#Function to create the unique BPMN_id
def create_BPMN_id(labels,pool_dict):

    BPMN_id = [class_dict[labels[i]] for i in range(len(labels))] 

    data_counter = 1

    enums = {
        'event': 1,
        'task': 1,
        'sequenceFlow': 1,
        'messageFlow': 1,
        'message_event': 1,
        'exclusiveGateway': 1,
        'parallelGateway': 1,
        'dataAssociation': 1,
        'pool': 1,
        'timerEvent': 1,
        'eventBasedGateway': 1
    }

    BPMN_name = [class_dict[label] for label in labels]

    for idx, Bpmn_id in enumerate(BPMN_name):
        key = {
            'event': 'event',
            'task': 'task',
            'dataObject': 'dataObject',
            'sequenceFlow': 'sequenceFlow',
            'messageFlow': 'messageFlow',
            'messageEvent': 'message_event',
            'exclusiveGateway': 'exclusiveGateway',
            'parallelGateway': 'parallelGateway',
            'dataAssociation': 'dataAssociation',
            'pool': 'pool',
            'dataStore': 'dataStore',
            'timerEvent': 'timerEvent',
            'eventBasedGateway': 'eventBasedGateway'
        }.get(Bpmn_id, None)

        if key:
            if key in ['dataObject', 'dataStore']:
                BPMN_id[idx] = f'{key}_{data_counter}'
                data_counter += 1
            else:
                BPMN_id[idx] = f'{key}_{enums[key]}'
                enums[key] += 1
        
    # Update the pool_dict keys with their corresponding BPMN_id values
    updated_pool_dict = {}
    for key, value in pool_dict.items():
        if key < len(BPMN_id):
            new_key = BPMN_id[key]
            updated_pool_dict[new_key] = value

    return BPMN_id, updated_pool_dict



def add_diagram_elements(parent, element_id, x, y, width, height):
    """Utility to add BPMN diagram notation for elements."""
    shape = ET.SubElement(parent, 'bpmndi:BPMNShape', attrib={
        'bpmnElement': element_id,
        'id': element_id + '_di'
    })
    bounds = ET.SubElement(shape, 'dc:Bounds', attrib={
        'x': str(x),
        'y': str(y),
        'width': str(width),
        'height': str(height)
    })

def add_diagram_edge(parent, element_id, waypoints):
    """Utility to add BPMN diagram notation for sequence flows."""
    edge = ET.SubElement(parent, 'bpmndi:BPMNEdge', attrib={
        'bpmnElement': element_id,
        'id': element_id + '_di'
    })
    for x, y in waypoints:
        if x is None or y is None:
            return
        ET.SubElement(edge, 'di:waypoint', attrib={
            'x': str(x),
            'y': str(y)
        })


def check_status(link, keep_elements):
    if link[0] in keep_elements and link[1] in keep_elements:
        return 'middle'
    elif link[0] is None and link[1] in keep_elements:
        return 'start'
    elif link[0] in keep_elements and link[1] is None:
        return 'end'
    else:
        return 'middle'
    
def check_data_association(i, links, labels, keep_elements):
    status, links_idx = [], []
    for j, (k,l) in enumerate(links):
        if labels[j] == list(class_dict.values()).index('dataAssociation'):
            if k==i:
                status.append('output')
                links_idx.append(j)
            elif l==i:
                status.append('input')
                links_idx.append(j)

    return status, links_idx

def create_data_Association(bpmn,data,size,element_id,current_idx,source_id,target_id):
    waypoints = calculate_waypoints(data, size, current_idx, source_id, target_id)
    add_diagram_edge(bpmn, element_id, waypoints)

def check_eventBasedGateway(i, links, labels):
    status, links_idx = [], []
    for j, (k,l) in enumerate(links):
        if labels[j] == list(class_dict.values()).index('sequenceFlow'):
            if k==i:
                status.append('output')
                links_idx.append(j)
            elif l==i:
                status.append('input')
                links_idx.append(j)

    return status, links_idx
        
# Function to dynamically create and layout BPMN elements
def create_bpmn_object(process, bpmnplane, text_mapping, definitions, size, data, keep_elements):
    elements = data['BPMN_id']
    positions = data['boxes']
    links = data['links']

    for i in keep_elements:
        if i >= len(elements):
            print("Problem with the index")
            continue
        element_id = elements[i]

        if element_id is None:
            continue
        
        element_type = element_id.split('_')[0]
        x, y = positions[i][:2]

        # Start Event
        if element_type == 'event':
            status = check_status(links[i], keep_elements)
            if status == 'start':
                element = ET.SubElement(process, 'bpmn:startEvent', id=element_id, name=text_mapping[element_id])
            elif status == 'middle':
                element = ET.SubElement(process, 'bpmn:intermediateCatchEvent', id=element_id, name=text_mapping[element_id])
            elif status == 'end':
                element = ET.SubElement(process, 'bpmn:endEvent', id=element_id, name=text_mapping[element_id])

            add_diagram_elements(bpmnplane, element_id, x, y, size['event'][0], size['event'][1])

        # Task
        elif element_type == 'task':
            element = ET.SubElement(process, 'bpmn:task', id=element_id, name=text_mapping[element_id])
            status, datasAssociation_idx = check_data_association(i, data['links'], data['labels'], keep_elements)

            if len(status) != 0:
                for state, dataAssociation_idx in zip(status, datasAssociation_idx):
                    # Handle Data Input Association
                    if state == 'input':
                        dataObject_idx = links[dataAssociation_idx][0]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        ET.SubElement(element, 'bpmn:property', id=f'Property_{dataAssociation_idx}_{dataObject_ref.split("_")[1]}', name='__targetRef_placeholder')
                        sub_element = ET.SubElement(element, 'bpmn:dataInputAssociation', id=f'dataInAsso_{dataAssociation_idx}_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:sourceRef').text = dataObject_ref
                        ET.SubElement(sub_element, 'bpmn:targetRef').text = f"Property_{dataAssociation_idx}_{dataObject_ref.split('_')[1]}"
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, dataObject_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        dataObject_idx = links[dataAssociation_idx][1]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        sub_element = ET.SubElement(element, 'bpmn:dataOutputAssociation', id=f'dataOutAsso_{dataAssociation_idx}_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:targetRef').text = dataObject_ref
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, element_id, dataObject_name)

            add_diagram_elements(bpmnplane, element_id, x, y, size['task'][0], size['task'][1])

        # Message Events (Start, Intermediate, End)
        elif element_type == 'message':
            status = check_status(links[i], keep_elements)
            if status == 'start':
                element = ET.SubElement(process, 'bpmn:startEvent', id=element_id, name=text_mapping[element_id])
            elif status == 'middle':
                element = ET.SubElement(process, 'bpmn:intermediateCatchEvent', id=element_id, name=text_mapping[element_id])
            elif status == 'end':
                element = ET.SubElement(process, 'bpmn:endEvent', id=element_id, name=text_mapping[element_id])

            status, datasAssociation_idx = check_data_association(i, data['links'], data['labels'], keep_elements)
            if len(status) != 0:
                for state, dataAssociation_idx in zip(status, datasAssociation_idx):
                    # Handle Data Input Association
                    if state == 'input':
                        dataObject_idx = links[dataAssociation_idx][0]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        sub_element = ET.SubElement(element, 'bpmn:dataInputAssociation', id=f'dataInAsso_{dataAssociation_idx}_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:sourceRef').text = dataObject_ref
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, dataObject_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        dataObject_idx = links[dataAssociation_idx][1]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        sub_element = ET.SubElement(element, 'bpmn:dataOutputAssociation', id=f'dataOutAsso_{dataAssociation_idx}_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:targetRef').text = dataObject_ref
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, element_id, dataObject_name)

            ET.SubElement(element, 'bpmn:messageEventDefinition', id=f'MessageEventDefinition_{i+1}')
            add_diagram_elements(bpmnplane, element_id, x, y, size['message'][0], size['message'][1])

        # Gateways (Exclusive, Parallel)
        elif element_type in ['exclusiveGateway', 'parallelGateway']:
            gateway_type = 'exclusiveGateway' if element_type == 'exclusiveGateway' else 'parallelGateway'
            element = ET.SubElement(process, f'bpmn:{gateway_type}', id=element_id)
            add_diagram_elements(bpmnplane, element_id, x, y, size[element_type][0], size[element_type][1])

        elif element_type == 'eventBasedGateway':
            element = ET.SubElement(process, 'bpmn:eventBasedGateway', id=element_id)
            status, links_idx = check_eventBasedGateway(i, data['links'], data['labels'])

            if len(status) != 0:
                for state, link_idx in zip(status, links_idx):
                    # Handle Data Input Association
                    if state == 'input' :
                        gateway_idx = links[link_idx][0]
                        gateway_name = elements[gateway_idx]
                        sub_element = ET.SubElement(element, 'bpmn:eventBasedGateway', id=f'eventBasedGateway_{link_idx}_{gateway_name.split("_")[1]}')
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], i, gateway_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        gateway_idx = links[link_idx][1]
                        gateway_name = elements[gateway_idx]
                        sub_element = ET.SubElement(element, 'bpmn:eventBasedGateway', id=f'eventBasedGateway_{link_idx}_{gateway_name.split("_")[1]}')
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], i, element_id, gateway_name)


            add_diagram_elements(bpmnplane, element_id, x, y, size['eventBasedGateway'][0], size['eventBasedGateway'][1])

        # Data Object
        elif element_type == 'dataObject' or element_type == 'dataStore':
            #print('ici dataObject', element_id)
            dataObject_idx = element_id.split('_')[1]
            dataObject_ref = f'DataObjectReference_{dataObject_idx}'
            if element_type == 'dataObject':
                ET.SubElement(process, 'bpmn:dataObjectReference', id=dataObject_ref, dataObjectRef=element_id, name=text_mapping[element_id])
                ET.SubElement(process, f'bpmn:{element_type}', id=element_id)
            elif element_type == 'dataStore':
                ET.SubElement(process, 'bpmn:dataStoreReference', id=dataObject_ref, name=text_mapping[element_id])
            add_diagram_elements(bpmnplane, dataObject_ref, x, y, size[element_type][0], size[element_type][1])

        # Timer Event
        elif element_type == 'timerEvent':
            element = ET.SubElement(process, 'bpmn:intermediateCatchEvent', id=element_id, name=text_mapping[element_id])
            ET.SubElement(element, 'bpmn:timerEventDefinition', id=f'TimerEventDefinition_{i+1}')
            add_diagram_elements(bpmnplane, element_id, x, y, size['timerEvent'][0], size['timerEvent'][1])


def calculate_pool_bounds(boxes, labels, keep_elements, size):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for i in keep_elements:
        if i >= len(labels):
            print("Problem with the index")
            continue
        
        element = labels[i]
        if element in {None, 7, 13, 14, 15}: 
            continue
        

        if size == None:
            element_width = boxes[i][2] - boxes[i][0]
            element_height = boxes[i][3] - boxes[i][1]
        else:
            element_width, element_height = size[class_dict[labels[i]]]
        
        x, y = boxes[i][:2]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + element_width)
        max_y = max(max_y, y + element_height)
    
    return min_x-50, min_y-50, max_x+50, max_y+50


    
def calculate_pool_waypoints(idx, data, size, source_idx, target_idx, source_element, target_element):
    # Get the bounding boxes of the source and target elements
    source_box = data['boxes'][source_idx]
    target_box = data['boxes'][target_idx]

    # Get the midpoints of the source element
    source_mid_x = (source_box[0] + source_box[2]) / 2
    source_mid_y = (source_box[1] + source_box[3]) / 2

    # Check if the connection involves a pool
    if source_element == 'pool':
        if target_element == 'pool':
            return [(source_mid_x, source_mid_y), (source_mid_x, source_mid_y)]
        pool_box = source_box
        element_box = (target_box[0], target_box[1], target_box[0]+size[target_element][0], target_box[1]+size[target_element][1])
        element_mid_x = (element_box[0] + element_box[2]) / 2
        element_mid_y = (element_box[1] + element_box[3]) / 2
        # Connect the pool's bottom or top side to the target element's top or bottom center
        if pool_box[3] < element_box[1]:  # Pool is above the target element
            waypoints = [(element_mid_x, pool_box[3]), (element_mid_x, element_box[1])]
        else:  # Pool is below the target element
            waypoints = [(element_mid_x, element_box[3]), (element_mid_x, pool_box[1])]
    else:
        pool_box = target_box
        element_box = (source_box[0], source_box[1], source_box[0]+size[source_element][0], source_box[1]+size[source_element][1])
        element_mid_x = (element_box[0] + element_box[2]) / 2
        element_mid_y = (element_box[1] + element_box[3]) / 2

        # Connect the element's bottom or top center to the pool's top or bottom side
        if pool_box[3] < element_box[1]:  # Pool is above the target element
            waypoints = [(element_mid_x, element_box[1]), (element_mid_x, pool_box[3])]
        else:  # Pool is below the target element
            waypoints = [(element_mid_x, element_box[3]), (element_mid_x, pool_box[1])]

    return waypoints

def add_curve(waypoints, pos_source, pos_target, threshold=30):
    """
    Add a single curve to the sequence flow by introducing a control point.
    The control point is added at an offset from the midpoint of the original waypoints.
    """
    if len(waypoints) < 2:
        return waypoints

    # Extract start and end points
    start_point = waypoints[0]
    end_point = waypoints[1]

    start_x, start_y = start_point
    end_x, end_y = end_point

    pos_horizontal = ['left', 'right']
    pos_vertical = ['top', 'bottom']

    if abs(start_x - end_x) < threshold or abs(start_y - end_y) < threshold:
        return waypoints

    # Calculate the control point
    if pos_source in pos_horizontal and pos_target in pos_horizontal:
        control_point = None
    elif pos_source in pos_vertical and pos_target in pos_vertical:
        control_point = None
    elif pos_source in pos_horizontal and pos_target in pos_vertical:
        control_point = (end_x, start_y)
    elif pos_source in pos_vertical and pos_target in pos_horizontal:
        control_point = (start_x, end_y)
    else:
        control_point = None
    

    # Create the curved path
    if control_point is not None:
        curved_waypoints = [start_point, control_point, end_point]
    else:
        curved_waypoints = [start_point, end_point]

    return curved_waypoints


def calculate_waypoints(data, size, current_idx, source_id, target_id):
    best_points = data['best_points'][current_idx]
    pos_source = best_points[0]
    pos_target = best_points[1]

    source_idx = data['BPMN_id'].index(source_id)
    target_idx = data['BPMN_id'].index(target_id)

    if source_idx == target_idx:
        warning()
        return None

    if source_idx is None or target_idx is None:
        warning()
        return None

    name_source = source_id.split('_')[0]
    name_target = target_id.split('_')[0]

    # Get the position of the source and target
    source_x, source_y = data['boxes'][source_idx][:2]
    target_x, target_y = data['boxes'][target_idx][:2]

    if name_source == 'pool' or name_target == 'pool':
        warning()
        return [(source_x, source_y), (target_x, target_y)]

    if pos_source == 'left':
        source_x = source_x
        source_y += size[name_source][1] / 2
    elif pos_source == 'right':
        source_x += size[name_source][0]
        source_y += size[name_source][1] / 2
    elif pos_source == 'top':
        source_x += size[name_source][0] / 2
        source_y = source_y
    elif pos_source == 'bottom':
        source_x += size[name_source][0] / 2
        source_y += size[name_source][1]

    if pos_target == 'left':
        target_x = target_x
        target_y += size[name_target][1] / 2
    elif pos_target == 'right':
        target_x += size[name_target][0]
        target_y += size[name_target][1] / 2
    elif pos_target == 'top':
        target_x += size[name_target][0] / 2
        target_y = target_y
    elif pos_target == 'bottom':
        target_x += size[name_target][0] / 2
        target_y += size[name_target][1]

    waypoints = [(source_x, source_y), (target_x, target_y)]

    # Add curve if no obstacles are in the path
    if data['labels'][current_idx] == list(class_dict.values()).index('sequenceFlow'):
        curved_waypoints = add_curve(waypoints, pos_source, pos_target)
    else:
        curved_waypoints = waypoints

    return curved_waypoints


def create_flow_element(bpmn, text_mapping, idx, size, data, parent, message=False):  
    source_idx, target_idx = data['links'][idx]

    if source_idx is None or target_idx is None:
        warning()
        return

    source_id, target_id = data['BPMN_id'][source_idx], data['BPMN_id'][target_idx]
    if message:
        element_id = f'messageflow_{source_id}_{target_id}'
    else:
        element_id = f'sequenceflow_{source_id}_{target_id}'

    if message:
        if source_id.split('_')[0] == 'pool' or target_id.split('_')[0] == 'pool':
            waypoints = calculate_pool_waypoints(idx, data, size, source_idx, target_idx, source_id.split('_')[0], target_id.split('_')[0])
            if source_id.split('_')[0] == 'pool':
                XML_source_id = f"participant_{source_id.split('_')[1]}"
                XML_target_id = target_id
            if target_id.split('_')[0] == 'pool':
                XML_target_id = f"participant_{target_id.split('_')[1]}"
                XML_source_id = source_id

            element = ET.SubElement(parent, 'bpmn:messageFlow', id=element_id, sourceRef=XML_source_id, targetRef=XML_target_id, name=text_mapping[data['BPMN_id'][idx]])
        else:
            waypoints = calculate_waypoints(data, size, idx, source_id, target_id)
            if waypoints is None:
                return
            element = ET.SubElement(parent, 'bpmn:messageFlow', id=element_id, sourceRef=source_id, targetRef=target_id, name=text_mapping[data['BPMN_id'][idx]])
    else:
        waypoints = calculate_waypoints(data, size, idx, source_id, target_id)
        if waypoints is None:
            return
        element = ET.SubElement(parent, 'bpmn:sequenceFlow', id=element_id, sourceRef=source_id, targetRef=target_id, name=text_mapping[data['BPMN_id'][idx]])
    add_diagram_edge(bpmn, element_id, waypoints)
    
    

