import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_hub as hub

model_path = "efficientnet_model.h5"  # Replace with your model path
# Function to load the model and make predictions
def predict_single_image(model_path, image_path):
    # Define a custom_objects dictionary to handle the KerasLayer from TensorFlow Hub
    custom_objects = {'KerasLayer': hub.KerasLayer}

    # Load the model from the HDF5 file with custom_objects
    model = load_model(model_path, custom_objects=custom_objects)

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Assuming the model input size is 224x224
    image = image.convert("RGB")
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values

    # Reshape the image to match the model's input shape
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)

    # Assuming it's a binary classification, return the class and probability
    class_label = "Tumor" if predictions[0][0] >= 0.5 else "No Tumor"
    probability = predictions[0][0]

    return class_label, probability


def brain_cancer_page():
    st.title("Brain Tumor Detection System")
    uploaded_file = st.file_uploader("Upload Brain MRI image here...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        predict_button = st.button("ㅤㅤPredictㅤㅤ")
        if predict_button:
            class_label, probability = predict_single_image(model_path, uploaded_file)
            probability*=100
            st.info(f"""
                    ##### Predicted Class: **{class_label}**
                    ##### Confidence: {probability:.2f}%
                    """)


CANCER_CLASS_LABELS = ['Benign', 'Malignant', 'Normal']
CANCER_MODEL_PATH = 'cancer_model.h5'
def preprocess_and_predict(model, class_labels, image_file, target_size, color_mode=None, scale_factor=1.0):
    img = image.load_img(image_file, target_size=target_size, color_mode=color_mode)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / scale_factor
    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]
    confidence = round(100 * np.max(pred), 2)
    return predicted_class, confidence

def cancer_page():
    st.title("Lung Cancer Detection System")
    uploaded_file = st.file_uploader("Upload chest x-ray image here...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        model = tf.keras.models.load_model(CANCER_MODEL_PATH)
        predict_button = st.button("ㅤㅤPredictㅤㅤ")
        if predict_button:
            predicted_class, confidence = preprocess_and_predict(model, CANCER_CLASS_LABELS, uploaded_file, (256, 256), 'grayscale', 255.0)
            st.info(f"""
                    ##### Predicted Class: **{predicted_class}**
                    ##### Confidence: {confidence}%
                    """)
st.set_page_config(
    page_title="Fatal Disease Prediction",
    page_icon=":microscope:"
)



selected=option_menu(
        menu_title=None,
        options=["Home","About Us","Contact","Model-1","Model-2"],
        icons=["house","book","envelope","lungs","asterisk"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

if selected == "Home":
        page_bg_img = '''
        <style>
        [data-testid="stAppViewContainer"]{
        background-image:url("https://wallpapercave.com/dwp2x/wp12404646.png"); 
         
        background-attachment: fixed;
        }body{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #EADB23;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.markdown('<p class="title-text"><span style="font-size: 60px;"><b>Predicting Fatal Diseases</b></span></p>', unsafe_allow_html=True)
        st.subheader("' Empowering Early Detection for a Healthier Tomorrow'")
        st.write("""<span style="font-size: 25px;"><i><b>Our mission is to provide cutting-edge solutions for early disease detection, optimal resource allocation, and informed decison-making, ultimately improving patient outcomes and advancing medical knowledge.</b></i></span>""",unsafe_allow_html=True)
        st.write("""<span style="font-size:20px;"><b>Blog:</b></span>""",unsafe_allow_html=True)      
        st.write("""<span style="font-size:20px;"><b><u>Introduction:</u></b>In our quest to conquer the world of medicine, one of the most pressing challenges has been the prediction of rare and fatal diseases. These conditions often strike unexpectedly, leaving both patients and healthcare providers with little time to react. However, recent advancements in medical research and technology have opened new doors for early detection and intervention. In this blog post, we'll explore some of the remarkable research and studies that have paved the way for a brighter future in predicting rare and fatal diseases.</span>""",unsafe_allow_html=True)
        st.write("""<span style="font-size:20px;"><b><u>Artificial Intelligence and Machine Learning:</u></b>Artificial intelligence (AI) and machine learning (ML) have become indispensable tools in the realm of disease prediction. Researchers are using AI algorithms to analyze vast datasets and identify hidden patterns in medical records, radiological images, and genetic information.For instance, AI-driven algorithms have shown incredible promise in predicting rare and fatal diseases like certain types of cancer and neurodegenerative disorders. These algorithms can recognize subtle changes in medical images, detect anomalies in lab results, and even predict disease progression based on a patient's historical data.Furthermore, AI is being used to develop predictive models that can assess the risk of developing a rare disease based on genetic markers, lifestyle factors, and environmental influences. This holistic approach to disease prediction offers a comprehensive understanding of an individual's risk factors.</span>""",unsafe_allow_html=True)
        st.write("""<span style="font-size:20px;"><b><u>Collaborative Research Initiatives:</u></b>The fight against rare and fatal diseases is a global endeavor. International collaborations and research initiatives have become crucial in accelerating the prediction and understanding of these conditions. Organizations like the World Health Organization (WHO) and national health agencies are supporting research efforts, sharing data, and funding studies focused on rare diseases.Collaboration is essential in pooling resources, knowledge, and expertise. By working together, researchers can access larger and more diverse datasets, leading to more accurate predictive models and early detection methods.</span>""",unsafe_allow_html=True)
        st.write("""<span style="font-size:20px;"><b><u>Conclusion:</u></b>The prediction of rare and fatal diseases is no longer a distant dream. Thanks to genomics, AI ML, and collaborative research efforts, we are making great strides in identifying individuals at risk and providing timely interventions. As research in this field continues to advance, we can look forward to a future where early detection of rare and fatal diseases becomes the norm rather than the exception. With these tools at our disposal, we are closer than ever to unlocking the secrets of disease prediction and prevention, bringing hope and relief to millions around the world.





</span>""",unsafe_allow_html=True)




if selected == "About Us":
        page_bg_color = '''
        <style>
        [data-testid="stAppViewContainer"]{
        background-image:url("https://wallpapercave.com/dwp2x/wp12404646.png"); 
        background-attachment: fixed;
        }body{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #EADB23;
        }
        </style>
        '''
        st.markdown(page_bg_color, unsafe_allow_html=True)

        st.title(f"About Us")
        st.subheader(f"We are a group of passionate B.Tech Computer Science students who share a common vision—to harness the potential of technology to transform healthcare and save lives.")
        st.write("""
        <span style="font-size: 20px;">Our journey began as a project driven by curiosity and a desire to apply our technical skills to address critical health challenges. As computer science enthusiasts, we embarked on a mission to develop a powerful predictive system that can assist individuals in understanding their health risks early on. With a deep-rooted belief in the transformative power of data and machine learning, we've created a platform that leverages cutting-edge algorithms to analyze health data and provide personalized insights. Our aim is to make advanced disease prediction accessible to everyone, regardless of their medical expertise. We want to empower individuals to take proactive steps towards better health, armed with the knowledge and tools we provide. While we may not have medical degrees, our dedication to improving healthcare outcomes through technology is unwavering. Join us in this exciting journey as we use our computer science skills to pave the way for a healthier future.</span>
        """, unsafe_allow_html=True)
        


if selected == "Contact":
        page_bg_color = '''
        <style>
        [data-testid="stAppViewContainer"]{
        background-image:url("https://wallpapercave.com/dwp2x/wp12404646.png"); 
        background-attachment: fixed;
        }body{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #EADB23;
        }
        </style>
        '''
        st.markdown(page_bg_color, unsafe_allow_html=True)        
        st.header(":mailbox: Get in touch with us!")
        contact_form="""
        <form action="https://formsubmit.co/atul2004v@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Type your message here:"></textarea>
        <button type="submit">Send</button>
        </form>
        """
        st. markdown(contact_form, unsafe_allow_html=True)
        
        def local_css(file_name):
                with open(file_name) as f:
                        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                
        local_css("style/style.css")




if selected == "Model-1":
        page_bg_color = '''
        <style>
        [data-testid="stAppViewContainer"]{
        background-image:url("https://wallpapercave.com/dwp2x/wp12404646.png");  
        background-attachment: fixed;
        }body{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #EADB23;
        }
        </style>
        '''
        st.markdown(page_bg_color, unsafe_allow_html=True)
        cancer_page()
        st.write("""<span style="font-size: 20px;"><b>Lung cancer is one of the most prevalent and deadliest forms of cancer worldwide. It primarily affects the lungs, where abnormal cell growth leads to the formation of tumors. There are two main types: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC), each with its own subtypes. Lung cancer is often asymptomatic in its early stages, which contributes to its high fatality rate. Common symptoms, such as persistent cough, chest pain, and shortness of breath, usually appear when the cancer has already advanced.</b></span>""",unsafe_allow_html=True)
   



if selected == "Model-2":
        page_bg_color = '''
        <style>
        [data-testid="stAppViewContainer"]{
        background-image:url("https://wallpapercave.com/dwp2x/wp12404646.png");  
        background-attachment: fixed;
        }body{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #EADB23;
        }
        </style>
        '''
        st.markdown(page_bg_color, unsafe_allow_html=True)           
        brain_cancer_page()
        st.write("""<span style="font-size: 20px;"><b>A brain tumor is an abnormal growth of cells within the brain or its surrounding tissues. It can be either benign (non-cancerous) or malignant (cancerous), and both types can cause various neurological symptoms. The exact cause of brain tumors is often unknown, but they can result from genetic factors, exposure to radiation, or other environmental factors. Common symptoms of brain tumors include headaches, seizures, changes in vision, memory problems, and difficulties with motor skills. Early diagnosis and treatment are crucial for managing brain tumors, and treatment options may include surgery, radiation therapy, chemotherapy, or a combination of these approaches. The prognosis and treatment plan depend on the type, location, and stage of the tumor, as well as the patient's overall health.</b></span>""",unsafe_allow_html=True)



    
