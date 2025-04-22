import streamlit as st
from io import BytesIO
from PIL import Image
import keras
import numpy as np
import tensorflow as tf



st.markdown("<h1 style='text-align: center; color: #0C6F0F;'>Potato Diseases Classification</h1>", unsafe_allow_html=True)


#change the background
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             
             background-color:#EDFBEB ;
             background-attachment: fixed;
             background-size: cover;
             text-align:center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()


new_image = st.file_uploader(label='Upload your file',type='jpeg')         # upload file

model = keras.models.load_model('plant6.keras')     #load the model
model.save('plant6.keras', overwrite=True)
model = keras.models.load_model('plant6.keras')



CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]    

def read_file_as_image(data) -> np.ndarray:           # read the file as image
    image = np.array((Image.open(BytesIO(data)).resize((256, 256))))
    return image

def predict(
    file
):
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = model.predict(img_batch)
    

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100
    confidence = round(confidence,2)
    return predicted_class, confidence

def img():
    if new_image != None:
        st.image(new_image,caption="Input Image",width=600)
img()



if st.button('predict'): 
    ha = predict(new_image)
   
    st.markdown(f"<h1 style='text-align: center; color: #0C6F0F;'> Result     : {ha[0]} </h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #0C6F0F;'> Confidence : {ha[1]}% </h1>", unsafe_allow_html=True)
