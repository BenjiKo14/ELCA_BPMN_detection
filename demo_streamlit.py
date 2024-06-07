import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance

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
        'start': (36, 36),
        'task': (100, 80),
        'message': (36, 36),
        'messageEvent': (36, 36),
        'end': (36, 36),
        'exclusiveGateway': (50, 50),
        'event': (36, 36),
        'parallelGateway': (50, 50),
        'sequenceFlow': (150, 10),
        'pool': (250, 100),
        'lane': (200, 100),
        'dataObject': (40, 60),
        'dataStore': (60, 60),
        'subProcess': (120, 90),
        'eventBasedGateway': (50, 50),
        'timerEvent': (40, 40),
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

def display_bpmn_xml(bpmn_xml):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>BPMN Modeler</title>
        <link rel="stylesheet" href="https://unpkg.com/bpmn-js/dist/assets/diagram-js.css">
        <link rel="stylesheet" href="https://unpkg.com/bpmn-js/dist/assets/bpmn-font/css/bpmn-embedded.css">
        <script src="https://unpkg.com/bpmn-js/dist/bpmn-modeler.development.js"></script>
        <style>
            html, body {{
                height: 100%;
                padding: 0;
                margin: 0;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
            }}
            #button-container {{
                padding: 10px;
                background-color: #ffffff;
                border-bottom: 1px solid #ddd;
                display: flex;
                justify-content: flex-start;
                gap: 10px;
            }}
            #save-button, #download-button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }}
            #download-button {{
                background-color: #008CBA;
            }}
            #canvas-container {{
                flex: 1;
                position: relative;
            }}
            #canvas {{
                height: 100%;
                width: 100%;
                position: absolute;
            }}
        </style>
    </head>
    <body>
        <div id="button-container">
            <button id="save-button">Save BPMN</button>
            <button id="download-button">Download XML</button>
        </div>
        <div id="canvas-container">
            <div id="canvas"></div>
        </div>
        <script>
            var bpmnModeler = new BpmnJS({{
                container: '#canvas'
            }});

            async function openDiagram(bpmnXML) {{
                try {{
                    await bpmnModeler.importXML(bpmnXML);
                    bpmnModeler.get('canvas').zoom('fit-viewport');
                    bpmnModeler.get('canvas').zoom(0.8); // Adjust this value for zooming out
                }} catch (err) {{
                    console.error('Error rendering BPMN diagram', err);
                }}
            }}

            async function saveDiagram() {{
                try {{
                    const result = await bpmnModeler.saveXML({{ format: true }});
                    const xml = result.xml;
                    const blob = new Blob([xml], {{ type: 'text/xml' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'diagram.bpmn';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }} catch (err) {{
                    console.error('Error saving BPMN diagram', err);
                }}
            }}

            async function downloadXML() {{
                const xml = `{bpmn_xml}`;
                const blob = new Blob([xml], {{ type: 'text/xml' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'diagram.xml';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}

            document.getElementById('save-button').addEventListener('click', saveDiagram);
            document.getElementById('download-button').addEventListener('click', downloadXML);

            openDiagram(`{bpmn_xml}`);
        </script>
    </body>
    </html>
    """
    components.html(html_template, height=1000, width=1500)



# Function to load the models only once and use session state to keep track of it
#@st.cache_resource
def load_model():
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.prediction_up = False

    if not st.session_state.model_loaded:
        opti_name = 'Adam'
        model_to_load = 'model_AdamW_60ep_4batch_trainval_blur00_crop01_flip01_rotate02_only_arrow6_withkey'
        model_arrow, _, _ = prepare_model(arrow_dict, opti_name, model_to_load=model_to_load, model_type='arrow')
        model_to_load = 'model_AdamW_30ep_4batch_trainval_blur02_crop03_flip02_rotate02_only_object2'
        model_object, _, _ = prepare_model(object_dict, opti_name, model_to_load=model_to_load, model_type='object')
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
    st.session_state.text_mapping = mapping_text(st.session_state.prediction, st.session_state.text_pred, print_sentences=True, percentage_thresh=0.5)
                
    # Remove the original image display
    image_placeholder.empty()

def main():
    st.set_page_config(layout="wide")
    st.title("BPMN model recognition demo")

    # Initialize the session state for storing pool bounding boxes
    if 'pool_bboxes' not in st.session_state:
        st.session_state.pool_bboxes = []

    # Load the model using the defined function
    load_model()
    prediction_up = st.session_state.prediction_up
    model_arrow = st.session_state.model_arrow
    model_object = st.session_state.model_object

    #Create the layout for the app
    col1, col2 = st.columns(2)
    with col1:
        # Create a file uploader for the user to upload an image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image if the user has uploaded an image
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
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
                    st.success('Detection completed!')
                    st.session_state.prediction_up = True

    # If the prediction has been made and the user has uploaded an image, display the options for the user to annotate the image
    if 'prediction' in st.session_state and uploaded_file is not None:
        display_options(st.session_state.crop_image, score_threshold)

        if st.session_state.prediction_up==True:
            st.session_state.bpmn_xml = create_XML(st.session_state.prediction.copy(), st.session_state.text_mapping, st.session_state.scale)
            st.session_state.prediction_up = False
    
        display_bpmn_xml(st.session_state.bpmn_xml)

if __name__ == "__main__":
    main()
