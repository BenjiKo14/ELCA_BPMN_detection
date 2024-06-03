import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from torchvision.transforms import functional as F
from eval import full_prediction, object_prediction
from OCR import text_prediction, filter_text, mapping_text, rescale
from train import prepare_model
from utils import draw_annotations, create_loader, class_dict, arrow_dict, object_dict
from toXML import create_BPMN_id, calculate_pool_bounds, add_diagram_elements
from pathlib import Path
from toXML import create_bpmn_elements, create_flow_element
import xml.etree.ElementTree as ET
import numpy as np
from display import draw_stream
from streamlit_image_comparison import image_comparison
from xml.dom import minidom
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from train import get_faster_rcnn_model, get_arrow_model
import gdown

def read_xml_file(filepath):
    """ Read XML content from a file """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def create_XML(full_pred, text_mapping):
    namespaces = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'di': 'http://www.omg.org/spec/DD/20100524/DI',
        'dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    size = {
    'start': (36, 36),
    'task': (100, 80),
    'message': (36, 36),
    'end': (36, 36),
    'exclusiveGateway': (50, 50),
    'event': (36, 36),
    'parallelGateway': (50, 50),
    'sequenceFlow': (150, 10),  # Added
    'pool': (250, 100),  # Added
    'lane': (200, 100),  # Added
    'dataObject': (60, 60),  # Added
    'dataAssociation': (150, 10),  # Added
    'dataStore': (60, 60),  # Added
    'subProcess': (120, 90),  # Added
    'eventBasedGateway': (50, 50),  # Added
    'timerEvent': (40, 40),  # Added
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
    collaboration = ET.SubElement(definitions, 'bpmn:collaboration', id='collaboration_1')
    bpmndi = ET.SubElement(definitions, 'bpmndi:BPMNDiagram', id='BPMNDiagram_1')
    bpmnplane = ET.SubElement(bpmndi, 'bpmndi:BPMNPlane', id='BPMNPlane_1', bpmnElement='collaboration_1')

    full_pred['boxes'] = rescale(st.session_state.scale,full_pred['boxes'])

    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        pool_id = f'participant_{idx+1}'
        pool = ET.SubElement(collaboration, 'bpmn:participant', id=pool_id, processRef=f'process_{idx+1}', name=text_mapping[full_pred['BPMN_id'][list(full_pred['pool_dict'].keys())[idx]]])
        
        # Calculate the bounding box for the pool
        if len(keep_elements) == 0:
            print('No elements in pool')
            min_x, min_y, max_x, max_y = full_pred['boxes'][pool_index]
            pool_width = max_x - min_x
            pool_height = max_y - min_y
        else:
            min_x, min_y, max_x, max_y = calculate_pool_bounds(full_pred, keep_elements, size)
            pool_width = max_x - min_x + 100  # Adding padding
            pool_height = max_y - min_y + 100  # Adding padding
        
        add_diagram_elements(bpmnplane, pool_id, min_x - 50, min_y - 50, pool_width, pool_height)


    # Create BPMN elements for each pool
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        create_bpmn_elements(idx, bpmnplane, text_mapping, definitions, size, full_pred, keep_elements)


    #create message flow
    message_flows = [i for i, label in enumerate(full_pred['labels']) if class_dict[label] == 'messageFlow']
    for idx in message_flows:
        create_flow_element(bpmnplane, text_mapping, idx, size, full_pred, collaboration, message=True)


    # Create flow elements
    for idx, (pool_index, keep_elements) in enumerate(full_pred['pool_dict'].items()):
        for i in keep_elements:
            if full_pred['labels'][i] == list(class_dict.values()).index('sequenceFlow'):
                create_flow_element(bpmnplane, text_mapping, i, size, full_pred, definitions[idx + 2], message=False)
    

    tree = ET.ElementTree(definitions)
    rough_string = ET.tostring(definitions, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    full_pred['boxes'] = rescale(1/st.session_state.scale,full_pred['boxes'])

    return pretty_xml_as_string

def display_XML(bpmn_xml):
    # Using bpmn-js Modeler
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BPMN Modeler</title>
        <script src="https://unpkg.com/bpmn-js/dist/bpmn-modeler.production.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #f4f4f4;
            }}
            #canvas {{
                width: 100%;
                height: 100vh;  // Adjust if necessary
                border: none;
            }}
        </style>
    </head>
    <body>
        <div id="canvas"></div>
        <script>
            var bpmnModeler = new BpmnJS({{
                container: '#canvas',
                keyboard: {{
                    bindTo: window
                }}
            }});

            async function openDiagram(bpmnXML) {{
                try {{
                    await bpmnModeler.importXML(bpmnXML);
                    console.log('Success! BPMN 2.0 diagram successfully rendered.');
                    bpmnModeler.get('canvas').zoom('fit-viewport', 'auto');  // Auto-adjusts the view to fit the diagram
                    bpmnModeler.get('canvas').zoom(bpmnModeler.get('canvas').zoom() * 0.8); // Zooms out to 80% of the fit view
                }} catch (err) {{
                    console.error('Failed to render diagram.', err);
                }}
            }}

            // Load and open BPMN diagram
            openDiagram(`{bpmn_xml}`);

            // Save the diagram when user has done the editing
            function saveDiagram() {{
                bpmnModeler.saveXML({{ format: true }}, function(err, xml) {{
                    if (err) {{
                        console.log('Error saving BPMN 2.0 diagram', err);
                    }} else {{
                        console.log('Diagram saved', xml);
                        // Here you could potentially send the XML back to the server or elsewhere
                    }}
                }});
            }}

            window.addEventListener('beforeunload', saveDiagram);
        </script>
    </body>
    </html>
    """

    components.html(html_template, height=1000, width=1500)

def load_model():
    """Load the model only once, and use session state to keep track of it."""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    if not st.session_state.model_loaded:        
        model_object = get_faster_rcnn_model(len(object_dict))
        model_arrow = get_arrow_model(len(arrow_dict),2)

        url_arrow = 'https://drive.google.com/uc?id=1xwfvo7BgDWz-1jAiJC1DCF0Wp8YlFNWt'
        url_object = 'https://drive.google.com/uc?id=1GiM8xOXG6M6r8J9HTOeMJz9NKu7iumZi'

        # Define paths to save models
        output_arrow = 'model_arrow.pth'
        output_object = 'model_object.pth'

        # Download models using gdown
        gdown.download(url_arrow, output_arrow, quiet=False)
        gdown.download(url_object, output_object, quiet=False)

        # Load models
        with st.spinner('Loading model...'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_arrow.load_state_dict(torch.load(output_arrow, map_location=device))
            model_object.load_state_dict(torch.load(output_object, map_location=device))
        st.session_state.model_loaded = True
        st.session_state.model_arrow = model_arrow
        st.session_state.model_object = model_object

from PIL import Image, ImageEnhance
def prepare_image(image, pad=True, new_size=(1333, 1333)):
    # Preprocess the image
    

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
        image = F.pad(image, padding, fill=200, padding_mode='constant')

    
    return new_scaled_size, image


def display_options(image,score_threshold):
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
    
    annotated_image = draw_stream(
        np.array(image), prediction=st.session_state.prediction, text_predictions=st.session_state.text_pred,
        draw_keypoints=draw_keypoints, draw_boxes=draw_boxes, draw_links=draw_links, draw_twins=False, draw_grouped_text=draw_text,
        write_class=write_class, write_text=write_text, keypoints_correction=True, write_idx=write_idx,
        score_threshold=score_threshold, write_score=write_score, resize=True, return_image=True, axis=True
    )

    image_comparison(
        img1=annotated_image,
        img2=image,
        label1="Annotated Image",
        label2="Original Image",
        starting_position=99,
        width=1000,
    )

def perform_inference(model_object, model_arrow, image, score_threshold):
    #progress_placeholder = st.empty()
    #message_placeholder = st.empty()

    _, uploaded_image = prepare_image(image, pad=False)
              
    img_tensor = F.to_tensor(prepare_image(image.convert('RGB'))[1])

    # Progress bar setup
    #my_bar = progress_placeholder.progress(0)
    #message_placeholder.text('Loading model and predicting...')

    # Display original image
    if 'image_placeholder' not in st.session_state:
        image_placeholder = st.empty()  # Create an empty placeholder
    image_placeholder.image(uploaded_image, caption='Original Image', width=1000)


    # Prediction
    _, st.session_state.prediction = full_prediction(model_object, model_arrow, img_tensor, score_threshold=score_threshold, iou_threshold=0.5)

    ocr_results = text_prediction(uploaded_image)

    st.session_state.text_pred = filter_text(ocr_results, threshold=0.4)
    #BPMN_id = set(st.session_state.prediction['BPMN_id'])  # This ensures uniqueness of task names
    #st.session_state.text_mapping = {id: '' for id in BPMN_id} 
    st.session_state.text_mapping = mapping_text(st.session_state.prediction, st.session_state.text_pred, print_sentences=False)
                
    
    # Remove progress bar and message
    #progress_placeholder.empty()
    image_placeholder.empty()
    #message_placeholder.empty()

def main():
    st.set_page_config(layout="wide")
    st.title("BPMN model recognition demo")

    # Initialize the session state for storing pool bounding boxes
    if 'pool_bboxes' not in st.session_state:
        st.session_state.pool_bboxes = []

    # Load the model using the defined function
    load_model()

    # Now, you can access the models from session_state anywhere in your app
    model_arrow = st.session_state.model_arrow
    model_object = st.session_state.model_object

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        with col1:           
            cropped_image = st_cropper(original_image, realtime_update=True, box_color='#0000FF', should_resize_image=True, default_coords=(30, original_image.size[0]-30, 30, original_image.size[1]-30))
        with col2:
            st.image(cropped_image, caption="Cropped Image", use_column_width=False, width=500)
            
 

        if cropped_image is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                score_threshold = st.slider("Set score threshold for prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            with col2:
                st.session_state.scale = st.slider("Set scale for XML file", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
                

            if st.button("Launch Prediction"):
                st.session_state.crop_image = cropped_image
                with st.spinner('Processing...'):
                    perform_inference(model_object, model_arrow, st.session_state.crop_image, score_threshold)
                st.success('Detection completed!')

    if 'prediction' in st.session_state and uploaded_file is not None:
        display_options(st.session_state.crop_image,score_threshold)
        st.session_state.bpmn_xml = create_XML(st.session_state.prediction, st.session_state.text_mapping)
        st.download_button(label="Download XML", data=st.session_state.bpmn_xml, file_name="bpmn_model.xml", mime='text/xml')
        st.download_button(label="Download BPMN File", data=st.session_state.bpmn_xml, file_name="bpmn_model.bpmn", mime='text/xml')
        display_XML(st.session_state.bpmn_xml)


if __name__ == "__main__":
    print("Starting Streamlit app...")
    main()
