import streamlit as st
import boto3
import json

#%%writefile app.py

def make_prediction(image_bytes):
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName='https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/jumpstart-example-FT-tensorflow-ic-imag-2023-10-19-03-49-36-994/invocations',
                                       ContentType = 'application/x-image',
                                       Region = 'us-east-1',
                                       Body=image_bytes)
    result = json.loads(response['Body'].read().decode())
    return result

st.title('Gram Stain Image Classifier')
uploaded_file = st.file_uploader('Choose an image to classify…', type = ['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    result = make_prediction(image_bytes)
    label = {result['predicted_label']}
    st.write("###")
    st.write("Classifying...")
    #st.write(f'Predicticted label: {result['predicted_label']}')
    #st.write(f'Probability: {result['probability']})
    if label == "GN":
        st.write("The Gram stain contains Gram negative bacteria")
        st.write(f"Probability: {result['probability']}")

    else:
        st.write("The Gram stain contains Gram positive bacteria")
        st.write(f"Probability: {result['probability']}")
