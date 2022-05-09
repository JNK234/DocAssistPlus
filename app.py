# Imports 
import streamlit as st      
import pydicom

from matplotlib.pyplot import sca
from PIL import Image
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from fastai.medical.imaging import PILDicom

import Constants

##############################################################################################################################
########################################## UTILS #############################################################################
##############################################################################################################################

def load_img(img_file):
    img = Image.open(img_file)
    return img

def get_display_text(scan_type):
    if scan_type == 'MRI':
        return Constants.MRI_DISPLAY_TEXT, ["jpg", "png", "jpeg"]
    elif scan_type == 'X-Ray':
        return Constants.XRAY_DISPLAY_TEXT, ["dcm"]

def image_upload(scan_type):
    displayText, fileTypes = get_display_text(scan_type)
    uploaded_file = st.file_uploader(displayText, type=fileTypes)

    if uploaded_file is not None:
        if scan_type == 'MRI':
            img = load_img(uploaded_file)
            return img, PILImage.create(uploaded_file)
        elif scan_type == 'X-Ray':
            dicomFile = pydicom.dcmread(uploaded_file, force=True)
            img = PILImage.create(dicomFile.pixel_array)
            dicomFile.save_as("Test.dcm")
            return img, Path("Test.dcm")

    else:
        return None, None


# Define getters for SIIM X-ray inference results
def get_x(x):
    return siim_small_path/x['file']

def get_y(x):
    return x['label']

# Function to perform SIIM X-ray inference and return results
def inference(learner_path, file_path):
    learner = load_learner(learner_path)
    inference_class, _ ,inference_prob = learner.predict(file_path)
    print(f"{Constants.INTERNAL_LOG}Inference class: {inference_class}")
    print(f"{Constants.INTERNAL_LOG}Inference probability: {inference_prob}")
    return inference_class, inference_prob.max() * 100

def get_predictions(scan_type, file):
    return inference(Constants.MODEL_SCAN_MAPPING[scan_type], file)

##############################################################################################################################
########################################## Streamlit Widgets #################################################################
##############################################################################################################################

# Necessary Display widgets
st.markdown(f"<h1 style='text-align: center; color: white;'>{Constants.TITLE_TEXT}</h1>", unsafe_allow_html=True) 
st.subheader(f"{Constants.SUBHEADER_TEXT_1}")
st.write(f"{Constants.DESCRIPTION_TEXT_1}")

st.subheader(f"{Constants.SUBHEADER_TEXT_2}")
st.write(f"{Constants.DESCRIPTION_TEXT_2}")

st.subheader(f"{Constants.SUBHEADER_TEXT_3}")

# Get the Patient's name
patientName = st.text_input("Please enter Patient's name: ")
if patientName != "":
    print(f"{Constants.INTERNAL_LOG}Patient's name: {patientName}")

# Selection Dropdown menu 
option = st.selectbox(                
      'Please select the scan type from the dropdown menu: ',
     ('X-Ray', 'MRI'))
print(f"{Constants.INTERNAL_LOG}Scan type: {option}")

# Display the selected san option
st.write('Scan type Selected:', option)

# Upload the image code
st.subheader("Upload the scan")
scan_img, uploadFile = image_upload(scan_type=option)

# Display the uploaded image
if scan_img is not None:
    # Process and Display the image
    st.subheader("Processing the Scan")

    st.write(f"{Constants.PROCESSING_TEXT}")

    st.text("Uploaded file:")
    print(f"{Constants.INTERNAL_LOG}Type of uploaded file:", type(scan_img))

    # Display the image
    st.image(scan_img, width=250, caption=f"Uploaded {option} Scan")

    # Find the Predictions 
    inference_class, inference_probabity = get_predictions(scan_type=option, file=uploadFile)

    # Display the inference results
    st.subheader("Inference Results")
    st.write(f"""The model has identified the ailment as:
    {inference_class}
    with a probability of {inference_probabity}%""")



