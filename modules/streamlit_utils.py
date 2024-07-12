import streamlit as st
from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import functional as F
import gc
import psutil
import numpy as np
from pathlib import Path
import gdown
import os

from modules.OCR import text_prediction, filter_text, mapping_text
from modules.utils import class_dict, arrow_dict, object_dict
from modules.display import draw_stream
from modules.eval import full_prediction
from modules.train import get_faster_rcnn_model, get_arrow_model
from streamlit_image_comparison import image_comparison

from streamlit_image_annotation import detection
from modules.toXML import create_XML
from modules.eval import develop_prediction, generate_data
from modules.utils import class_dict, object_dict

from modules.htlm_webpage import display_bpmn_xml
from streamlit_cropper import st_cropper
from streamlit_image_select import image_select
from streamlit_js_eval import streamlit_js_eval

from modules.toWizard import create_wizard_file
from huggingface_hub import hf_hub_download
import time




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




# Suppress the symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Function to load the models only once and use session state to keep track of it
def load_models():
    with st.spinner('Loading model...'):
        model_object = get_faster_rcnn_model(len(object_dict))
        model_arrow = get_arrow_model(len(arrow_dict), 2)

        model_arrow_path = hf_hub_download(repo_id="BenjiELCA/BPMN_Detection", filename="model_arrow.pth")
        model_object_path = hf_hub_download(repo_id="BenjiELCA/BPMN_Detection", filename="model_object.pth")

        # Define paths to save models
        output_arrow = 'model_arrow.pth'
        output_object = 'model_object.pth'

        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model arrow
        if not Path(output_arrow).exists():
            # Download model from Hugging Face Hub
            model_arrow.load_state_dict(torch.load(model_arrow_path, map_location=device))
            st.session_state.model_arrow = model_arrow
            print('Model arrow downloaded from Hugging Face Hub')
            # Save the model locally
            torch.save(model_arrow.state_dict(), output_arrow)
        elif 'model_arrow' not in st.session_state and Path(output_arrow).exists():
            model_arrow.load_state_dict(torch.load(output_arrow, map_location=device))
            st.session_state.model_arrow = model_arrow
            print('Model arrow loaded from local file')
 

        # Load model object
        if not Path(output_object).exists():
            # Download model from Hugging Face Hub
            model_object.load_state_dict(torch.load(model_object_path, map_location=device))
            st.session_state.model_object = model_object
            print('Model object downloaded from Hugging Face Hub')
            # Save the model locally
            torch.save(model_object.state_dict(), output_object)
        elif 'model_object' not in st.session_state and Path(output_object).exists():
            model_object.load_state_dict(torch.load(output_object, map_location=device))
            st.session_state.model_object = model_object
            print('Model object loaded from local file')


        # Move models to device
        model_arrow.to(device)
        model_object.to(device)

        # Update session state
        st.session_state.model_loaded = True

        return model_object, model_arrow

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
        image = enhancer.enhance(1.0)  # Adjust the brightness if necessary
        # Pad the resized image to make it exactly the desired size
        padding = [0, 0, new_size[0] - new_scaled_size[0], new_size[1] - new_scaled_size[1]]
        image = F.pad(image, padding, fill=200, padding_mode='edge')

    return image

# Function to display various options for image annotation
def display_options(image, score_threshold, is_mobile, screen_width):
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
        np.array(image), prediction=st.session_state.original_prediction, text_predictions=st.session_state.text_pred,
        draw_keypoints=draw_keypoints, draw_boxes=draw_boxes, draw_links=draw_links, draw_twins=False, draw_grouped_text=draw_text,
        write_class=write_class, write_text=write_text, keypoints_correction=True, write_idx=write_idx, only_show=selected_option,
        score_threshold=score_threshold, write_score=write_score, resize=True, return_image=True, axis=True
    )

    if is_mobile is True:
        width = screen_width
    else:
        width = screen_width//2

    # Display the original and annotated images side by side
    image_comparison(
        img1=annotated_image,
        img2=image,
        label1="Annotated Image",
        label2="Original Image",
        starting_position=99,
        width=width,
    )

# Function to perform inference on the uploaded image using the loaded models
def perform_inference(model_object, model_arrow, image, score_threshold, is_mobile, screen_width, iou_threshold=0.5, distance_treshold=30, percentage_text_dist_thresh=0.5):
    uploaded_image = prepare_image(image, pad=False)
              
    img_tensor = F.to_tensor(prepare_image(image.convert('RGB')))

    # Display original image
    if 'image_placeholder' not in st.session_state:
        image_placeholder = st.empty()  # Create an empty placeholder
    if is_mobile is False:
        width = screen_width
        if is_mobile is False:
            width = screen_width//2
        image_placeholder.image(uploaded_image, caption='Original Image', width=width)

    # Prediction
    _, st.session_state.prediction = full_prediction(model_object, model_arrow, img_tensor, score_threshold=score_threshold, iou_threshold=iou_threshold, distance_treshold=distance_treshold)

    # Perform OCR on the uploaded image
    ocr_results = text_prediction(uploaded_image)

    # Filter and map OCR results to prediction results
    st.session_state.text_pred = filter_text(ocr_results, threshold=0.6)
    st.session_state.text_mapping = mapping_text(st.session_state.prediction, st.session_state.text_pred, print_sentences=False, percentage_thresh=percentage_text_dist_thresh)
                
    # Remove the original image display
    image_placeholder.empty()

    # Force garbage collection
    gc.collect()

    return image, st.session_state.prediction, st.session_state.text_mapping

@st.cache_data
def get_image(uploaded_file):
    return Image.open(uploaded_file).convert('RGB')


def configure_page():
    st.set_page_config(layout="wide")
    screen_width = streamlit_js_eval(js_expressions='screen.width', want_output=True, key='SCR')
    is_mobile = screen_width is not None and screen_width < 800
    return is_mobile, screen_width

def display_banner(is_mobile):
    # JavaScript expression to detect dark mode
    dark_mode_js = """
    (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
    """
    
    # Evaluate JavaScript in Streamlit to check for dark mode
    is_dark_mode = streamlit_js_eval(js_expressions=dark_mode_js, key='dark_mode')

    if is_mobile:
        if is_dark_mode:
            st.image("./images/banner_mobile_dark.png", use_column_width=True)
        else:
            st.image("./images/banner_mobile.png", use_column_width=True)
    else:
        if is_dark_mode:
            st.image("./images/banner_desktop_dark.png", use_column_width=True)
        else:
            st.image("./images/banner_desktop.png", use_column_width=True)

def display_title(is_mobile):
    title = "Welcome on the BPMN AI model recognition app"
    if is_mobile:
        title = "Welcome on the mobile version of BPMN AI model recognition app"
    st.title(title)

def display_sidebar():
    st.sidebar.header("This BPMN AI model recognition is proposed by: \n ELCA in collaboration with EPFL.")
    st.sidebar.subheader("Instructions:")
    st.sidebar.text("1. Upload you image")
    st.sidebar.text("2. Crop the image \n  (try to put the BPMN diagram \n   in the center of the image)")
    st.sidebar.text("3. Set the score threshold \n   for prediction (default is 0.5)")
    st.sidebar.text("4. Click on 'Launch Prediction'")
    st.sidebar.text("5. You can now see the annotation \n   and the BPMN XML result")
    st.sidebar.text("6. You can modify the result \n   by clicking on:\n   'Method and Style modification'")
    st.sidebar.text("7. You can change the scale for \n   the XML file and the size of \n   elements (default is 1.0)")
    st.sidebar.text("8. You can modify with modeler \n   and download the result in \n   right format")

    st.sidebar.subheader("If there is an error, try to:")
    st.sidebar.text("1. Change the score threshold")
    st.sidebar.text("2. Re-crop the image by placing\n   the BPMN diagram in the center\n   of the image")
    st.sidebar.text("3. Re-Launch the prediction")

    st.sidebar.subheader("You can close this sidebar")

def initialize_session_state():
    if 'pool_bboxes' not in st.session_state:
        st.session_state.pool_bboxes = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if not st.session_state.model_loaded:
        clear_memory()
        load_models()
        st.rerun()

def load_example_image():
    with st.expander("Use example images"):
        img_selected = image_select(
            "If you have no image and just want to test the demo, click on one of these images", 
            ["./images/none.jpg", "./images/example1.jpg", "./images/example2.jpg", "./images/example3.jpg", "./images/example4.jpg"],
            captions=["None", "Example 1", "Example 2", "Example 3", "Example 4"], 
            index=0, 
            use_container_width=False, 
            return_value="original"
        )
        return img_selected

def load_user_image(img_selected, is_mobile):
    if img_selected == './images/none.jpg':
        img_selected = None

    if img_selected is not None:
        uploaded_file = img_selected
    else:
        if is_mobile:
            uploaded_file = st.file_uploader("Choose an image from my computer...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        else:
            col1, col2 = st.columns(2)
            with col1:
                uploaded_file = st.file_uploader("Choose an image from my computer...", type=["jpg", "jpeg", "png"])

    return uploaded_file

def display_image(uploaded_file, screen_width, is_mobile):
    
    with st.spinner('Waiting for image display...'):
        original_image = get_image(uploaded_file)
        resized_image = original_image.resize((screen_width // 2, int(original_image.height * (screen_width // 2) / original_image.width)))

        if not is_mobile:
            cropped_image = crop_image(resized_image, original_image)
        else:
            st.image(resized_image, caption="Image", use_column_width=False, width=int(4/5 * screen_width))
            cropped_image = original_image

    return cropped_image

def crop_image(resized_image, original_image):
    marge = 10
    cropped_box = st_cropper(
        resized_image,
        realtime_update=True,
        box_color='#0000FF',
        return_type='box',
        should_resize_image=False,
        default_coords=(marge, resized_image.width - marge, marge, resized_image.height - marge)
    )
    scale_x = original_image.width / resized_image.width
    scale_y = original_image.height / resized_image.height
    x0, y0, x1, y1 = int(cropped_box['left'] * scale_x), int(cropped_box['top'] * scale_y), int((cropped_box['left'] + cropped_box['width']) * scale_x), int((cropped_box['top'] + cropped_box['height']) * scale_y)
    cropped_image = original_image.crop((x0, y0, x1, y1))
    return cropped_image

def get_score_threshold(is_mobile):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.score_threshold = st.slider("Set score threshold for prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.05) 

def launch_prediction(cropped_image, score_threshold, is_mobile, screen_width):
    st.session_state.crop_image = cropped_image
    with st.spinner('Processing...'):
        image, _ , _ = perform_inference(
            st.session_state.model_object, st.session_state.model_arrow, st.session_state.crop_image,
            score_threshold, is_mobile, screen_width, iou_threshold=0.3, distance_treshold=30, percentage_text_dist_thresh=0.5
        )
        st.balloons()    
        return image
    

def modify_results(percentage_text_dist_thresh=0.5):
    with st.expander("Method and Style modification"):
        label_list = list(object_dict.values())
        if st.session_state.prediction['labels'][-1] == 6:
            bboxes = [[int(coord) for coord in box] for box in st.session_state.prediction['boxes'][:-1]]
            labels = [int(label) for label in st.session_state.prediction['labels'][:-1]]
        else:
            bboxes = [[int(coord) for coord in box] for box in st.session_state.prediction['boxes']]
            labels = [int(label) for label in st.session_state.prediction['labels']]
        for i in range(len(bboxes)):
            bboxes[i][2] = bboxes[i][2] - bboxes[i][0]
            bboxes[i][3] = bboxes[i][3] - bboxes[i][1]

        arrow_bboxes = st.session_state.arrow_pred['boxes']
        arrow_labels = st.session_state.arrow_pred['labels']
        arrow_score = st.session_state.arrow_pred['scores']
        arrow_keypoints = st.session_state.arrow_pred['keypoints']

        # Filter boxes and labels where label is less than 12 to only have objects
        object_bboxes = []
        object_labels = []     
        for i in range(len(bboxes)):
            if labels[i] <= 12:
                object_bboxes.append(bboxes[i])
                object_labels.append(labels[i])

        uploaded_image = prepare_image(st.session_state.crop_image, new_size=(1333, 1333), pad=False)

        new_data = detection(
            image=uploaded_image, bboxes=object_bboxes, labels=object_labels, 
            label_list=label_list, line_width=3, width=2000, use_space=False
        )

        if new_data is not None:
            changes = False
            new_lab = np.array([data['label_id'] for data in new_data])  
            # Convert back to original format
            bboxes = np.array([data['bbox'] for data in new_data])
            object_bboxes = np.array(object_bboxes)

            # Order bboxes and labels
            order = np.argsort(bboxes[:, 0])
            bboxes = bboxes[order]
            new_lab = new_lab[order]

            order2 = np.argsort(object_bboxes[:, 0])
            object_bboxes = object_bboxes[order2]
            object_labels = np.array(object_labels)[order2]

            # Make all values of bboxes integers
            bboxes = bboxes.astype(int)

            tolerance = 1

            object_labels = np.array(object_labels)


            if len(object_bboxes) == len(bboxes):
                # Calculate absolute differences
                abs_diff = np.abs(object_bboxes - bboxes)
                
                for i in range(len(object_bboxes)):
                    for j in range(len(object_bboxes[i])):
                        if abs_diff[i][j] > tolerance:
                            changes = True
                            break

                #check if labels are the same
                if not np.array_equal(object_labels, new_lab):
                    changes = True
            else:   
                changes = True                

            for i in range(len(bboxes)):
                bboxes[i][2] = bboxes[i][2] + bboxes[i][0]
                bboxes[i][3] = bboxes[i][3] + bboxes[i][1]

            object_scores = []
            object_keypoints = []
            for i in range(len(new_data)):
                object_scores.append(1.0)
                object_keypoints.append([[0, 0, 0], [0, 0, 0]])

            new_bbox = np.concatenate((bboxes, arrow_bboxes))
            new_lab = np.concatenate((new_lab, arrow_labels))
            new_scores = np.concatenate((object_scores, arrow_score))
            new_keypoints = np.concatenate((object_keypoints, arrow_keypoints))

            
            boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict = develop_prediction(new_bbox, new_lab, new_scores, new_keypoints, class_dict, correction=True)

            st.session_state.prediction = generate_data(st.session_state.prediction['image'], boxes, labels, scores, keypoints, bpmn_id, flow_links, best_points, pool_dict)
            st.session_state.text_mapping = mapping_text(st.session_state.prediction, st.session_state.text_pred, print_sentences=False, percentage_thresh=percentage_text_dist_thresh)

            if changes:
                st.rerun()

            return True

        


def display_bpmn_modeler(is_mobile, screen_width):
    with st.spinner('Waiting for BPMN modeler...'):
        st.session_state.bpmn_xml = create_XML(
            st.session_state.prediction.copy(), st.session_state.text_mapping, 
            st.session_state.size_scale, st.session_state.scale
        )
        st.session_state.vizi_file = create_wizard_file(st.session_state.prediction.copy(), st.session_state.text_mapping)
        display_bpmn_xml(st.session_state.bpmn_xml, st.session_state.vizi_file,  is_mobile=is_mobile, screen_width=int(4/5 * screen_width))

def modeler_options(is_mobile):
    if not is_mobile:
        with st.expander("Options for BPMN modeler"):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.scale = st.slider("Set distance scale for XML file", min_value=0.1, max_value=2.0, value=1.0, step=0.1) 
                st.session_state.size_scale = st.slider("Set size object scale for XML file", min_value=0.5, max_value=2.0, value=1.0, step=0.1) 
    else:
        st.session_state.scale = 1.0
        st.session_state.size_scale = 1.0