import streamlit as st
from torchvision.transforms import functional as F
import gc
import numpy as np

from modules.streamlit_utils import *
from modules.utils import error


def main():
    # Example usage
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    st.session_state.first_run = True
    is_mobile, screen_width = configure_page()
    display_banner(is_mobile)
    display_title(is_mobile)
    display_sidebar()

    initialize_session_state()

    cropped_image = None

    img_selected = load_example_image()
    uploaded_file = load_user_image(img_selected, is_mobile)

    if uploaded_file is not None:
        cropped_image = display_image(uploaded_file, screen_width, is_mobile)

    if uploaded_file is not None:
        get_score_threshold(is_mobile)
    
        if st.button("🚀 Launch Prediction"):
            st.session_state.image = launch_prediction(cropped_image, st.session_state.score_threshold, is_mobile, screen_width)
            st.session_state.original_prediction = st.session_state.prediction.copy()
            st.rerun()

    # Create placeholders for all sections
    prediction_result_placeholder = st.empty()
    additional_options_placeholder = st.empty()
    modeler_placeholder = st.empty()


    if 'prediction' in st.session_state and uploaded_file:
        if st.session_state.image != cropped_image:
            print('Image has changed')
            # Delete the prediction
            del st.session_state.prediction
            return

        if len(st.session_state.prediction['labels'])==0:
            error("No prediction available. Please upload a BPMN image or decrease the detection score treshold.")
        else:
            with prediction_result_placeholder.container():
                if is_mobile:
                    display_options(st.session_state.crop_image, st.session_state.score_threshold, is_mobile, int(5/6*screen_width))
                else:
                    with st.expander("Show result of prediction"):
                        display_options(st.session_state.crop_image, st.session_state.score_threshold, is_mobile, int(5/6*screen_width))

            if not is_mobile:
                with additional_options_placeholder.container():
                    state = modify_results()

          
            with modeler_placeholder.container():
                modeler_options(is_mobile)
                display_bpmn_modeler(is_mobile, screen_width)
    else:
        prediction_result_placeholder.empty()
        additional_options_placeholder.empty()
        modeler_placeholder.empty()
        # Create a lot of space for scrolling
        for _ in range(50):
            st.text("")

    gc.collect()

if __name__ == "__main__":
    print('Starting the app...')
    main()
