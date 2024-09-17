import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

def Camera():
    st.write("Please use the camera below to capture an image.")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner('Analyzing...'):
            try:
                result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
                st.success("Analysis Complete!")

                if isinstance(result, list):
                    result = result[0]

                st.write(f"**Dominant Emotion:** {result['dominant_emotion'].capitalize()}")
                st.bar_chart(result['emotion'])
            except Exception as e:
                st.error(f"An error occurred: {e}")

        st.image(image, caption='Your Picture', use_column_width=True)
    else:
        st.write("Waiting for an image...")