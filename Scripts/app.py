# Libraries
import gc
from pytest import Session
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pickle
import os
from model import *
from feedback import *
import base64

# Functions

# On change and click functions
def tag_caption_change():
    st.session_state.tag_caption_ = True
    st.session_state.send_feedback_button_ = False
def predict_button_click():
    st.session_state.predict_button_ = True
    st.session_state.tag_caption_ = False
def uploaded_file_change():
    st.session_state.uploaded_file_ = True
    st.session_state.predict_button_ = False
    st.session_state.img_id_ += 1
def send_feedback_button_click():
    st.session_state.send_feedback_button_ = True
def init_ss():
    st.session_state.img_id_ = 0
    if 'uploaded_file_' not in st.session_state.keys():
        st.session_state.uploaded_file_ = False
def hash_g_u(gcp_user):
    base64.b64encode(str(gcp_user.credentials).encode('utf-8'))

# Cache functions
@st.cache(hash_funcs={
    GCP_USER: hash_g_u
}, allow_output_mutation=True)
def init_gcp_user(credentials):
    logging.info('LOADING GCP USER...')
    gcp_user = GCP_USER(credentials)
    return gcp_user

@st.cache(hash_funcs={
    ImageDescriptor: lambda _: json.load(open('Data/Objects/info_model.json'))["ID"]
}, allow_output_mutation=True)
def load_model(path_parameters, path_checkpoint, path_tokenizer):
    img_descriptor = ImageDescriptor()
    img_descriptor.parameters(path_parameters)
    img_descriptor.architecture(path_tokenizer)
    img_descriptor.checkpoint(path_checkpoint)
    return img_descriptor

# Title and headers
init_ss()
try:
    with open('Data/Objects/credentials.json') as json_file:
        credentials = json.load(json_file)
except:
    credentials = st.secrets["gcp_service_account"]
gcp_user = init_gcp_user(credentials)
st.title('Machine Learning Web App - Image Captioning')
st.header("Click [here](https://github.com/juanse1608/AST-ImageCaptioning/blob/main/README.md) to know more about the project!")

# Ask for an image
st.write('''Upload a photo and see the predicted caption for it''')
img_descriptor = load_model('Data/Objects/parameters.json', 'Data/Model/', 'Data/Objects/tokenizer.pickle')
uploaded_file = st.file_uploader(label="Upload Image", type=["png", "jpeg", "jpg"], key='uploaded_file', on_change=uploaded_file_change)
if st.session_state.uploaded_file_:
    # Read and save image
    st.session_state.uploaded_image = uploaded_file.read()
    st.image(st.session_state.uploaded_image, use_column_width=True)

    # Prediction
    st.subheader('Prediction Parameters')
    col1, _, _ = st.columns(3)
    predict_type = col1.selectbox("Select Prediction Type", ("Argmax", "Probabilistic"), help='''__Argmax__ picks the value/token with the highest probability.
    __Probabilistic__ picks the value/token randomly using the distribution of the predictions.''', key='predict_type')
    predict_button = st.button('Predict Caption', key='predict_button', on_click=predict_button_click)
    if st.session_state.predict_button or st.session_state.predict_button_:
        img, _ = img_descriptor.preprocess(st.session_state.uploaded_image)
        result, caption, attention_plot, _ = img_descriptor.predict(img, predict_type)
        st.subheader(caption + '.')
        fig = img_descriptor.plot_attention(result=result, attention_plot=attention_plot, path=st.session_state.uploaded_image)
        st.subheader('Neural Network Attention Word by Word')
        st.pyplot(fig)

        # Feedback
        st.session_state.feedback = SEND_FEEDBACK
        if st.session_state.feedback:
            st.subheader('Feedback')
            tag_caption = st.selectbox("Is this correct?", ("Select an option", "Yes", "No"), key='tag_caption', on_change=tag_caption_change)
            if st.session_state.tag_caption_:
                if tag_caption != 'Select an option':
                    if tag_caption == 'Yes':
                        correct_caption = caption
                    elif tag_caption == 'No':
                        correct_caption = st.text_input("What should the correct caption be?", "", key='correct_caption')
                    send_feedback_button = st.button('Send Feedback', key='send_feedback_button', on_click=send_feedback_button_click)
                    if st.session_state.send_feedback_button_:
                        gcp_user.load_image_and_caption(
                            bigquery_table_name = 'img-captioning-322620.feedbacks.CORRECT_CAPTIONS',
                            caption = correct_caption,
                            feedback_query = 'Data/Feedback/feedbacks_new_id.sql',
                            storage_bucket_name = 'img-captioning-feedback',
                            uploaded_image = st.session_state.uploaded_image
                        )
                        st.write("Thank you for that, we'll use your help to make our model better!")
                        



                    
                

                

        

