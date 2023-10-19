import streamlit as st
import boto3
import json

#%%writefile app.py

def make_prediction(image_bytes):
    runtime = boto3.Session().client('sagemaker-runtime', region_name= 'us-east-1')
    response = runtime.invoke_endpoint(EndpointName='jumpstart-example-FT-tensorflow-ic-imag-2023-10-19-03-49-36-994',
                                       ContentType = 'application/x-image',
                                       Body=image_bytes)
    result = json.loads(response['Body'].read().decode())
    return result

st.title('Gram Stain Image Classifier')
uploaded_file = st.file_uploader('Choose an image to classify…', type = ['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    result = make_prediction(image_bytes)
    st.write("---")
    st.write(result) #TESTING
    if result.probabilities[0] > result.probabilities[1]:
        bacteria_class = "GN"
        bacteria_class_name = "Gram Negative bacteria"
    else:
        bacteria_class = "GP"
        bacteria_class_name = "Gram Positive bacteria"

    
    st.write(f"The Gram stain contains {bacteria_class_name}")
    st.write(f"Probability of being in the {bacteria_class_name}: {probabilities[0] if bacteria_class == 'GN' else probabilities[1]}")

