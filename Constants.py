###############################################################################
# Text constants used by DocAssist+
###############################################################################
from tarfile import SUPPORTED_TYPES


TITLE_TEXT = "Welcome to DocAssist+"
SUBHEADER_TEXT_1 = "About DocAssist+"
SUBHEADER_TEXT_2 = "How to use DocAssist+"
SUBHEADER_TEXT_3 = "Get Started"

DESCRIPTION_TEXT_1 = """
DocAssist+ is an intelligent web application that can aid you in easily idenitfying the disease/ailment
given the medical scans of the patient. This app uses state-of-the-art deep neural network models to 
understand the medical scans and provide you with the best possible diagnosis.

In this app, we support two types of medical scans:

* MRI scans (Magnetic Resonance Imaging): 
These are the most common scans used to identify 
the disease/ailment in the brain. We have trained a convolutional neural network model to identify 
whether the given MRI scan is demented or not. Our model is capable to identify different stages of 
dementia i.e. No dementia, Very Mild dementia, Mild dementia and Moderate dementia.

* X-Ray scans (X-Ray): 
These are the most common scans used to identify the disease/ailment in 
the body. We have developed a model by training on the X-ray scans of patients suffering from Pneumothorax
and normal X-ray scans. This web would help you to easily identify whether the patient suffers
from Pneumothorax given the X-ray scan. 

Important feature of this app is that it accepts medical scans of different types i.e. an image of type
png, jpg, jpeg etc or a DICOM file.
"""

DESCRIPTION_TEXT_2 = """
First you should enter the patient's name. Then you should select the type of scan from the dropdown menu.
To use this app, you need to upload an image of type png, jpg, jpeg or dcm. In the case of MRI scans,
only the image of type png, jpg, jpeg will be accepted. In the case of X-Ray scans, only the DICOM file.

After you upload/drop the image, you will get the results of the inference. The results will have 
predicted class i.e. the ailement detected and how confident that the model is on its prediction.
"""
PROCESSING_TEXT = "The model is processing the scan. This may take a few seconds. Please wait..."

INFERENCE_TEXT = """
The model has identified the ailment as:
"""


DICOM_DESCRIPTION_TYPE = """
DICOM or Digital Imaging and Communications in Medicine is the standard for
the communication and management of medical imaging information and related data. DICOM is most 
commonly used for storing and transmitting medical images enabling the integration of medical 
imaging devices such as scanners, servers, workstations, printers, network hardware, and 
picture archiving and communication systems from multiple manufacturers.
"""

MRI_DISPLAY_TEXT = "Please upload the MRI scan file. Supported file formats: jpg, png, jpeg"
XRAY_DISPLAY_TEXT = "Please upload the X-Ray scan file. Supported file format: Dicom"

INTERNAL_LOG = "INTERNAL LOG: "

###############################################################################
# Path constants used by DocAssist+
###############################################################################

XRAY_LEARNER_PATH = "Models/siim_pneumothorax_classifier_resnet18.pkl"
MRI_LEARNER_PATH = "Models/mri_dementia_classifier-resnet34.pkl"

###############################################################################
# Data constants used by DocAssist+
###############################################################################

MODEL_SCAN_MAPPING = {"MRI": MRI_LEARNER_PATH, 
                    "X-Ray": XRAY_LEARNER_PATH}