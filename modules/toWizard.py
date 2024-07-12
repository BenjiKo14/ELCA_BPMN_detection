import xml.etree.ElementTree as ET
from modules.utils import class_dict
from xml.dom import minidom
from modules.utils import error

def rescale(scale, boxes):
    for i in range(len(boxes)):
        boxes[i] = [boxes[i][0] * scale, boxes[i][1] * scale, boxes[i][2] * scale, boxes[i][3] * scale]
    return boxes

def create_BPMN_id(data):
    enum_end, enum_start, enum_task, enum_sequence, enum_dataflow, enum_messflow, enum_messageEvent, enum_exclusiveGateway, enum_parallelGateway, enum_pool = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    BPMN_name = [class_dict[data['labels'][i]] for i in range(len(data['labels']))]
    for idx, Bpmn_id in enumerate(BPMN_name):
        if Bpmn_id == 'event':
            if data['links'][idx][0] is not None and data['links'][idx][1] is None:
                data['BPMN_id'][idx] = f'end_event_{enum_end}'
                enum_end += 1
            elif data['links'][idx][0] is None and data['links'][idx][1] is not None:
                data['BPMN_id'][idx] = f'start_event_{enum_start}'
                enum_start += 1
        elif Bpmn_id == 'task' or Bpmn_id == 'dataObject':
            data['BPMN_id'][idx] = f'task_{enum_task}'
            enum_task += 1
        elif Bpmn_id == 'sequenceFlow':
            data['BPMN_id'][idx] = f'sequenceFlow_{enum_sequence}'
            enum_sequence += 1
        elif Bpmn_id == 'messageFlow':
            data['BPMN_id'][idx] = f'messageFlow_{enum_messflow}'
            enum_messflow += 1
        elif Bpmn_id == 'messageEvent':
            data['BPMN_id'][idx] = f'message_event_{enum_messageEvent}'
            enum_messageEvent += 1
        elif Bpmn_id == 'exclusiveGateway':
            data['BPMN_id'][idx] = f'exclusiveGateway_{enum_exclusiveGateway}'
            enum_exclusiveGateway += 1
        elif Bpmn_id == 'parallelGateway':
            data['BPMN_id'][idx] = f'parallelGateway_{enum_parallelGateway}'
            enum_parallelGateway += 1
        elif Bpmn_id == 'dataAssociation':
            data['BPMN_id'][idx] = f'dataAssociation_{enum_sequence}'
            enum_dataflow += 1
        elif Bpmn_id == 'pool':
            data['BPMN_id'][idx] = f'pool_{enum_pool}'
            enum_pool += 1

    return data

def check_end(val):
    if val[1] is None:
        return True
    return False

def connect(data, text_mapping, i):
    target_idx = data['links'][i][1]  
    if target_idx >= len(data['links']):
        error('There is an error with the Vizi file, care when you download it.')
        return None, None
    current_id = data['BPMN_id'][i]
    next_idx = data['links'][target_idx][1]
    next_id = data['BPMN_id'][next_idx]
    next_text = text_mapping[next_id]
    current_text = text_mapping[current_id]

    return current_text, next_text

def check_start(val):
    if val[0] is None:
        return True
    return False



def create_wizard_file(data, text_mapping):
    root = ET.Element('methodAndStyleWizard')
    
    modelName = ET.SubElement(root, 'modelName')
    modelName.text = 'My Diagram'
    
    author = ET.SubElement(root, 'author')
    author.text = 'Benjamin'
    
    # Add pools to the collaboration element
    for idx, (pool_index, keep_elements) in enumerate(data['pool_dict'].items()):
        pool_id = f'participant_{idx+1}'
        pool = ET.SubElement(root, 'processName')
        pool.text = text_mapping[pool_index]
    
    processDescription = ET.SubElement(root, 'processDescription')


    for idx, Bpmn_id in enumerate(data['BPMN_id']):
        # Start Event
        element_type = Bpmn_id.split('_')[0]
        if element_type == 'message':
            eventType = 'Message'
        elif element_type == 'event':
            eventType = 'None'
        if idx >= len(data['links']):
            continue
        if check_start(data['links'][idx]) and (element_type=='event' or element_type=='message'):
            startEvent = ET.SubElement(root, 'startEvent', attrib={'name': text_mapping[Bpmn_id], 'eventType': eventType}) 
    
    requestMessage = ET.SubElement(root, 'requestMessage')
    requester = ET.SubElement(root, 'requester')
    
    endEvents = ET.SubElement(root, 'endStates')
    for idx, Bpmn_id in enumerate(data['BPMN_id']):
        # End States
        if idx >= len(data['links']):
            continue
        if check_end(data['links'][idx]) and Bpmn_id.split('_')[0] == 'event':
            if text_mapping[Bpmn_id] == '':
                text_mapping[Bpmn_id] = '(unnamed)'
            ET.SubElement(endEvents, 'endState', attrib={'name': text_mapping[Bpmn_id], 'eventType': 'None', 'isRegular': 'False'})
    
    
  
    activities = ET.SubElement(root, 'activities')
    
    for idx, activity_name in enumerate(data['BPMN_id']):
        if activity_name.startswith('task'):
            activity = ET.SubElement(activities, 'activity', attrib={'name': text_mapping.get(activity_name, activity_name), 'performer': ''})
            endStates = ET.SubElement(activity, 'endStates')
            current_text, next_text = connect(data, text_mapping, idx)
            if next_text is not None:
                ET.SubElement(endStates, 'endState', attrib={'name': next_text, 'isRegular': 'True'})
            ET.SubElement(activity, 'subActivities')
            ET.SubElement(activity, 'subActivityFlows')
            ET.SubElement(activity, 'messageFlows')
    
    activityFlows = ET.SubElement(root, 'activityFlows')
    i=0
    for i, link in enumerate(data['links']):
        if link[0] is None and link[1] is not None and (data['BPMN_id'][i].split('_')[0] == 'event' or data['BPMN_id'][i].split('_')[0] == 'message'):
            current_text, next_text = connect(data, text_mapping, i)
            if current_text is None or next_text is None:
                continue
            ET.SubElement(activityFlows, 'activityFlow', attrib={'startEvent': current_text, 'endState': '---', 'target': next_text, 'isMerging': 'False', 'isPredefined': 'True'})
            i+=1
        if link[0] is not None and link[1] is not None and data['BPMN_id'][i].split('_')[0] == 'task':
            current_text, next_text = connect(data, text_mapping, i)
            if current_text is None or next_text is None:
                continue
            ET.SubElement(activityFlows, 'activityFlow', attrib={'activity': current_text, 'endState': '---', 'target': next_text, 'isMerging': 'False', 'isPredefined': 'True'})
            i+=1
    
    ET.SubElement(root, 'participants')
    
    # Pretty print the XML
    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")
    
    return pretty_xml_str
