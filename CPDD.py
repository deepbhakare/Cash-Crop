#!/usr/bin/env python

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('v3_pred_cott_dis.h5')

# Define labels for prediction output
labels = {
    0: 'Alternaria Alternata',
    1: 'Anthracnose',
    2: 'Bacterial Blight',
    3: 'Corynespora Leaf Fall',
    4: 'Healthy',
    5: 'Grey Mildew'
}

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150,150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 150
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make prediction on input image
def predict(image):
    # Preprocess input image
    image = preprocess_image(image)
    # Make prediction using pre-trained model
    prediction = model.predict(image)
    # Convert prediction from probabilities to label
    label = labels[np.argmax(prediction)]
    # Return label and confidence score
    return label, prediction[0][np.argmax(prediction)]

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection [BETA]')
    # Set app description
    st.write('This app helps you to detect the type of disease in a cotton plant.')
    st.write('NOTE- This model only works on Cotton Plant. (Its under development, which predicts the exact disease of plant)')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make prediction
        label, score = predict(image)
        # Check if the predicted label is a disease
        if label != 'Healthy':
            # Display prediction
            st.write('Prediction: {} (confidence score: {:.2%})'.format(label, score))
            # Provide information about the disease
            if label == 'Alternaria Alternata':
                st.write('The cotton plant is infected with Alternaria Alternata, a fungal disease that can cause leaf spot and boll rot. Treatment options include removing infected plant parts and using fungicides.')
            elif label == 'Anthracnose':
                st.write('The cotton plant is infected with Anthracnose, a fungal disease that can cause leaf spots, stem cankers, and boll rot. Treatment options include removing infected plant parts and using fungicides.')
            elif label == 'Bacterial Blight':
                st.write('The cotton plant is infected with Bacterial Blight, a bacterial disease that can cause water-soaked spots on leaves, stems, and bolls. Treatment options include removing infected plant parts and using bactericides.')
            elif label == 'Corynespora Leaf Fall':
                st.write('The cotton plant is infected with Corynespora Leaf Fall, a fungal disease that can cause leaf spots and defoliation. Treatment options include removing infected plant parts and using fungicides.')
            else:
                st.write('The cotton plant is infected with Grey Mildew, a fungal disease that can cause leaf spots and defoliation. Treatment options include removing infected plant parts and using fungicides.')
        else:
            st.write("The cotton plant is healthy. No treatment is needed.")

if __name__ == '__main__':
    main()
