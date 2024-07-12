import xml.etree.ElementTree as ET
from modules.utils import class_dict

def rescale(scale, boxes):
    for i in range(len(boxes)):
                boxes[i] = [boxes[i][0]*scale,
                            boxes[i][1]*scale,
                            boxes[i][2]*scale,
                            boxes[i][3]*scale]
    return boxes

def create_BPMN_id(data):
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
        'dataObject': 1,
        'dataStore': 1,
        'timerEvent': 1
    }

    BPMN_name = [class_dict[label] for label in data['labels']]

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
            'timerEvent': 'timerEvent'
        }.get(Bpmn_id, None)

        if key:
            data['BPMN_id'][idx] = f'{key}_{enums[key]}'
            enums[key] += 1

    return data



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
                        sub_element = ET.SubElement(element, 'bpmn:dataInputAssociation', id=f'dataInputAssociation_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:sourceRef').text = dataObject_ref
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, dataObject_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        dataObject_idx = links[dataAssociation_idx][1]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        sub_element = ET.SubElement(element, 'bpmn:dataOutputAssociation', id=f'dataOutputAssociation_{dataObject_ref.split("_")[1]}')
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
                        sub_element = ET.SubElement(element, 'bpmn:dataInputAssociation', id=f'dataInputAssociation_{dataObject_ref.split("_")[1]}')
                        ET.SubElement(sub_element, 'bpmn:sourceRef').text = dataObject_ref
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], dataAssociation_idx, dataObject_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        dataObject_idx = links[dataAssociation_idx][1]
                        dataObject_name = elements[dataObject_idx]
                        dataObject_ref = f'DataObjectReference_{dataObject_name.split("_")[1]}'
                        sub_element = ET.SubElement(element, 'bpmn:dataOutputAssociation', id=f'dataOutputAssociation_{dataObject_ref.split("_")[1]}')
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
                        sub_element = ET.SubElement(element, 'bpmn:eventBasedGateway', id=f'eventBasedGateway{gateway_name.split("_")[1]}')
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], i, gateway_name, element_id)

                    # Handle Data Output Association
                    elif state == 'output':
                        gateway_idx = links[link_idx][1]
                        gateway_name = elements[gateway_idx]
                        sub_element = ET.SubElement(element, 'bpmn:eventBasedGateway', id=f'eventBasedGateway{gateway_name.split("_")[1]}')
                        create_data_Association(bpmnplane, data, size, sub_element.attrib['id'], i, element_id, gateway_name)


            add_diagram_elements(bpmnplane, element_id, x, y, size['eventBasedGateway'][0], size['eventBasedGateway'][1])

        # Data Object
        elif element_type == 'dataObject' or element_type == 'dataStore':
            #print('ici dataObject', element_id)
            dataObject_idx = element_id.split('_')[1]
            dataObject_ref = f'DataObjectReference_{dataObject_idx}'
            element = ET.SubElement(process, 'bpmn:dataObjectReference', id=dataObject_ref, dataObjectRef=element_id, name=text_mapping[element_id])
            ET.SubElement(process, f'bpmn:{element_type}', id=element_id)
            add_diagram_elements(bpmnplane, dataObject_ref, x, y, size[element_type][0], size[element_type][1])

        # Timer Event
        elif element_type == 'timerEvent':
            element = ET.SubElement(process, 'bpmn:intermediateCatchEvent', id=element_id, name=text_mapping[element_id])
            ET.SubElement(element, 'bpmn:timerEventDefinition', id=f'TimerEventDefinition_{i+1}')
            add_diagram_elements(bpmnplane, element_id, x, y, size['timerEvent'][0], size['timerEvent'][1])


def calculate_pool_bounds(data, keep_elements, size):
    min_x = min_y = float('10000')
    max_x = max_y = float('0')
    
    for i in keep_elements:
        if i >= len(data['BPMN_id']):
            print("Problem with the index")
            continue
        element = data['BPMN_id'][i]
        if element is None or data['labels'][i] == 13 or data['labels'][i] == 14 or data['labels'][i] == 15 or data['labels'][i] == 7 or data['labels'][i] == 15: 
            continue
        
        element_type = element.split('_')[0]
        x, y = data['boxes'][i][:2]
        element_width, element_height = size[element_type]
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + element_width)
        max_y = max(max_y, y + element_height)
    
    return min_x, min_y, max_x, max_y

    
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
            waypoints = [(element_mid_x, pool_box[3]-50), (element_mid_x, element_box[1])]
        else:  # Pool is below the target element
            waypoints = [(element_mid_x, element_box[3]), (element_mid_x, pool_box[1]-50)]
    else:
        pool_box = target_box
        element_box = (source_box[0], source_box[1], source_box[0]+size[source_element][0], source_box[1]+size[source_element][1])
        element_mid_x = (element_box[0] + element_box[2]) / 2
        element_mid_y = (element_box[1] + element_box[3]) / 2

        # Connect the element's bottom or top center to the pool's top or bottom side
        if pool_box[3] < element_box[1]:  # Pool is above the target element
            waypoints = [(element_mid_x, element_box[1]), (element_mid_x, pool_box[3]-50)]
        else:  # Pool is below the target element
            waypoints = [(element_mid_x, element_box[3]), (element_mid_x, pool_box[1]-50)]

    return waypoints

def calculate_waypoints(data, size, current_idx, source_id, target_id):
    best_points = data['best_points'][current_idx]
    pos_source = best_points[0]
    pos_target = best_points[1]

    source_idx = data['BPMN_id'].index(source_id)
    target_idx = data['BPMN_id'].index(target_id)

    if source_idx==target_idx:
        #return [data['keypoints'][current_idx][0][:2], data['keypoints'][current_idx][1][:2]]
        return None

    if source_idx is None or target_idx is None:
        return [(source_x, source_y), (target_x, target_y)]


    name_source = source_id.split('_')[0]
    name_target = target_id.split('_')[0]

    #Get the position of the source and target
    source_x, source_y = data['boxes'][source_idx][:2]
    target_x, target_y = data['boxes'][target_idx][:2]

    if name_source == 'pool' or name_target == 'pool':
        return [(source_x, source_y), (target_x, target_y)]

    if pos_source == 'left':
        source_x = source_x
        source_y += size[name_source][1]/2
    elif pos_source == 'right':
        source_x += size[name_source][0]
        source_y += size[name_source][1]/2
    elif pos_source == 'top':
        source_x += size[name_source][0]/2
        source_y = source_y
    elif pos_source == 'bottom':
        source_x += size[name_source][0]/2
        source_y += size[name_source][1]

    if pos_target == 'left':
        target_x = target_x
        target_y += size[name_target][1]/2
    elif pos_target == 'right':
        target_x += size[name_target][0]
        target_y += size[name_target][1]/2
    elif pos_target == 'top':
        target_x += size[name_target][0]/2
        target_y = target_y
    elif pos_target == 'bottom':
        target_x += size[name_target][0]/2
        target_y += size[name_target][1]

    return [(source_x, source_y), (target_x, target_y)]

def create_flow_element(bpmn, text_mapping, idx, size, data, parent, message=False):  
    source_idx, target_idx = data['links'][idx]

    if source_idx is None or target_idx is None:
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
    
    

