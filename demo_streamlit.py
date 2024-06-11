import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
from htlm_webpage import display_bpmn_xml
import gc
import psutil

from OCR import text_prediction, filter_text, mapping_text, rescale
from train import prepare_model
from utils import draw_annotations, create_loader, class_dict, arrow_dict, object_dict
from toXML import calculate_pool_bounds, add_diagram_elements
from pathlib import Path
from toXML import create_bpmn_object, create_flow_element
import xml.etree.ElementTree as ET
import numpy as np
from display import draw_stream
from eval import full_prediction
from streamlit_image_comparison import image_comparison
from xml.dom import minidom
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from utils import find_closest_object
from train import get_faster_rcnn_model, get_arrow_model
import gdown

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

def clear_memory():
    st.session_state.clear()
    gc.collect()

# Function to read XML content from a file
def read_xml_file(filepath):
    """ Read XML content from a file """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Function to modify bounding box positions based on the given sizes
def modif_box_pos(pred, size):
    for i, (x1, y1, x2, y2) in enumerate(pred['boxes']):
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        label = class_dict[pred['labels'][i]]
        if label in size:
            pred['boxes'][i] = [center[0] - size[label][0] / 2, center[1] - size[label][1] / 2, center[0] + size[label][0] / 2, center[1] + size[label][1] / 2]
    return pred

# Function to create a BPMN XML file from prediction results
def create_XML(full_pred, text_mapping, scale):
    namespaces = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'di': 'http://www.omg.org/spec/DD/20100524/DI',
        'dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    size_elements = {
        'start': (54, 54),
        'task': (150, 120),
        'message': (54, 54),
        'messageEvent': (54, 54),
        'end': (54, 54),
        'exclusiveGateway': (75, 75),
        'event': (54, 54),
        'parallelGateway': (75, 75),
        'sequenceFlow': (225, 15),
        'pool': (375, 150),
        'lane': (300, 150),
        'dataObject': (60, 90),
        'dataStore': (90, 90),
        'subProcess': (180, 135),
        'eventBasedGateway': (75, 75),
        'timerEvent': (60, 60),
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

    # Create BPMN collaboration element
    collaboration = ET.SubElement(definitions, 'bpmn:collaboration', id='collaboration_1')

    # Create BPMN process elements
    process = []
    for idx in range(len(full_pred['pool_dict'].items())):
        process_id = f'process_{idx+1}'
        process.append(ET.SubElement(definitions, 'bpmn:process', id=process_id, isExecutable='false', name=text_mapping[full_pred['BPMN_id'][list(full_pred['pool_dict'].keys())[idx]]]))

    bpmndi = ET.SubElement(definitions, 'bpmndi:BPMNDiagram', id='BPMNDiagram_1')
    bpmnplane = ET.SubElement(bpmndi, 'bpmndi:BPMNPlane', id='BPMNPlane_1', bpmnElement='collaboration_1')

    full_pred['boxes'] = rescale(scale, full_pred['boxes'])

    # Add diagram elements for each pool
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        pool_id = f'participant_{idx+1}'
        pool = ET.SubElement(collaboration, 'bpmn:participant', id=pool_id, processRef=f'process_{idx+1}', name=text_mapping[full_pred['BPMN_id'][list(full_pred['pool_dict'].keys())[idx]]])
        
        # Calculate the bounding box for the pool
        if len(keep_elements) == 0:
            min_x, min_y, max_x, max_y = full_pred['boxes'][pool_index]
            pool_width = max_x - min_x
            pool_height = max_y - min_y
        else:
            min_x, min_y, max_x, max_y = calculate_pool_bounds(full_pred, keep_elements, size_elements)
            pool_width = max_x - min_x + 100  # Adding padding
            pool_height = max_y - min_y + 100  # Adding padding
        
        add_diagram_elements(bpmnplane, pool_id, min_x - 50, min_y - 50, pool_width, pool_height)

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
            if full_pred['labels'][i] == list(class_dict.values()).index('sequenceFlow'):
                create_flow_element(bpmnplane, text_mapping, i, size_elements, full_pred, process[idx], message=False)
    
    # Generate pretty XML string
    tree = ET.ElementTree(definitions)
    rough_string = ET.tostring(definitions, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    full_pred['boxes'] = rescale(1/scale, full_pred['boxes'])

    return pretty_xml_as_string


# Function to load the models only once and use session state to keep track of it
def load_models():
    with st.spinner('Loading model...'):     
        model_object = get_faster_rcnn_model(len(object_dict))
        model_arrow = get_arrow_model(len(arrow_dict),2)

        url_arrow = 'https://drive.google.com/uc?id=1xwfvo7BgDWz-1jAiJC1DCF0Wp8YlFNWt'
        url_object = 'https://drive.google.com/uc?id=1GiM8xOXG6M6r8J9HTOeMJz9NKu7iumZi'

        # Define paths to save models
        output_arrow = 'model_arrow.pth'
        output_object = 'model_object.pth'

        # Download models using gdown
        if not Path(output_arrow).exists():
            # Download models using gdown
            gdown.download(url_arrow, output_arrow, quiet=False)
        else:
            print('Model arrow downloaded from local')
        if not Path(output_object).exists():
            gdown.download(url_object, output_object, quiet=False)
        else:
            print('Model object downloaded from local')

        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_arrow.load_state_dict(torch.load(output_arrow, map_location=device))
        model_object.load_state_dict(torch.load(output_object, map_location=device))
        st.session_state.model_loaded = True
        st.session_state.model_arrow = model_arrow
        st.session_state.model_object = model_object

# Function to prepare the image for processing
def prepare_image(image, pad=True, new_size=(1333, 1333)):
    original_size = image.size
    # Calculate scale to fit the new size while maintaining aspect ratio
    scale = min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    new_scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    # Resize image to new scaled size
    image = F.resize(image, (new_scaled_size[1], new_scaled_size[0]))

    if pad:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)  # Adjust the brightness if necessary
        # Pad the resized image to make it exactly the desired size
        padding = [0, 0, new_size[0] - new_scaled_size[0], new_size[1] - new_scaled_size[1]]
        image = F.pad(image, padding, fill=200, padding_mode='edge')

    return new_scaled_size, image

# Function to display various options for image annotation
def display_options(image, score_threshold):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        write_class = st.toggle("Write Class", value=True)
        draw_keypoints = st.toggle("Draw Keypoints", value=True)
        draw_boxes = st.toggle("Draw Boxes", value=True)
    with col2:
        draw_text = st.toggle("Draw Text", value=False)
        write_text = st.toggle("Write Text", value=False)
        draw_links = st.toggle("Draw Links", value=False)
    with col3:
        write_score = st.toggle("Write Score", value=True)
        write_idx = st.toggle("Write Index", value=False)
    with col4:
        # Define options for the dropdown menu
        dropdown_options = [list(class_dict.values())[i] for i in range(len(class_dict))]
        dropdown_options[0] = 'all'
        selected_option = st.selectbox("Show class", dropdown_options)

    # Draw the annotated image with selected options
    annotated_image = draw_stream(
        np.array(image), prediction=st.session_state.prediction, text_predictions=st.session_state.text_pred,
        draw_keypoints=draw_keypoints, draw_boxes=draw_boxes, draw_links=draw_links, draw_twins=False, draw_grouped_text=draw_text,
        write_class=write_class, write_text=write_text, keypoints_correction=True, write_idx=write_idx, only_print=selected_option,
        score_threshold=score_threshold, write_score=write_score, resize=True, return_image=True, axis=True
    )

    # Display the original and annotated images side by side
    image_comparison(
        img1=annotated_image,
        img2=image,
        label1="Annotated Image",
        label2="Original Image",
        starting_position=99,
        width=1000,
    )

# Function to perform inference on the uploaded image using the loaded models
def perform_inference(model_object, model_arrow, image, score_threshold):
    _, uploaded_image = prepare_image(image, pad=False)
              
    img_tensor = F.to_tensor(prepare_image(image.convert('RGB'))[1])

    # Display original image
    if 'image_placeholder' not in st.session_state:
        image_placeholder = st.empty()  # Create an empty placeholder
    image_placeholder.image(uploaded_image, caption='Original Image', width=1000)

    # Prediction
    _, st.session_state.prediction = full_prediction(model_object, model_arrow, img_tensor, score_threshold=score_threshold, iou_threshold=0.5)

    # Perform OCR on the uploaded image
    ocr_results = text_prediction(uploaded_image)

    # Filter and map OCR results to prediction results
    st.session_state.text_pred = filter_text(ocr_results, threshold=0.5)
    st.session_state.text_mapping = mapping_text(st.session_state.prediction, st.session_state.text_pred, print_sentences=False, percentage_thresh=0.5)
                
    # Remove the original image display
    image_placeholder.empty()

    # Force garbage collection
    gc.collect()

@st.cache_data
def get_image(uploaded_file):
    return Image.open(uploaded_file).convert('RGB')

def main():
    st.set_page_config(layout="wide")
    st.title("BPMN model recognition demo")
    
     # Display current memory usage
    memory_usage = get_memory_usage()
    print(f"Current memory usage: {memory_usage:.2f} MB")

    # Initialize the session state for storing pool bounding boxes
    if 'pool_bboxes' not in st.session_state:
        st.session_state.pool_bboxes = []

    # Load the models using the defined function
    if 'model_object' not in st.session_state or 'model_arrow' not in st.session_state:
        clear_memory()
        load_models()

    model_arrow = st.session_state.model_arrow
    model_object = st.session_state.model_object

    #Create the layout for the app
    col1, col2 = st.columns(2)
    with col1:
        # Create a file uploader for the user to upload an image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image if the user has uploaded an image
    if uploaded_file is not None:
        original_image = get_image(uploaded_file)
        col1, col2 = st.columns(2)

        # Create a cropper to allow the user to crop the image and display the cropped image
        with col1:           
            cropped_image = st_cropper(original_image, realtime_update=True, box_color='#0000FF', should_resize_image=True, default_coords=(30, original_image.size[0]-30, 30, original_image.size[1]-30))
        with col2:
            st.image(cropped_image, caption="Cropped Image", use_column_width=False, width=500)
            
        # Display the options for the user to set the score threshold and scale
        if cropped_image is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                score_threshold = st.slider("Set score threshold for prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            with col2:
                st.session_state.scale = st.slider("Set scale for XML file", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

            # Launch the prediction when the user clicks the button    
            if st.button("Launch Prediction"):
                st.session_state.crop_image = cropped_image
                with st.spinner('Processing...'):
                    perform_inference(model_object, model_arrow, st.session_state.crop_image, score_threshold)
                    st.session_state.prediction = modif_box_pos(st.session_state.prediction, object_dict)                    
            
                    print('Detection completed!')


    # If the prediction has been made and the user has uploaded an image, display the options for the user to annotate the image
    if 'prediction' in st.session_state and uploaded_file is not None:
        st.success('Detection completed!')
        display_options(st.session_state.crop_image, score_threshold)

        #if st.session_state.prediction_up==True:
        st.session_state.bpmn_xml = create_XML(st.session_state.prediction.copy(), st.session_state.text_mapping, st.session_state.scale)
    
        display_bpmn_xml(st.session_state.bpmn_xml)

        # Force garbage collection after display
        gc.collect()

if __name__ == "__main__":
    print('Starting the app...')
    main()
