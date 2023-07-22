import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import cv2

st.image("klu.png", use_column_width=True)

st.markdown("""<p align='center' style='font-size:35px;'><b style='background-color:black; color:white;'>IBM</b>&nbsp - Innovation Center for Education</p>""",unsafe_allow_html=True)


st.markdown("<h1 align='center' style='color:red; font-size:50px;'><ins>IBM ICE DAY HACKATHON</ins></h1>",unsafe_allow_html=True)
st.markdown("<h2 align='center'' style='color:GREEN'>MEDICAL DIAGNOSIS</h2>",unsafe_allow_html=True)
st.write(" Develop a machine learning model that can accurately diagnose medical conditions from medical imaging data. Use deep learning algorithms to train the model on a large dataset of medical images and develop an accurate diagnosis model.")
st.markdown("---")

a=st.sidebar.radio("Select ",["Details","Fever","Sugar","Malaria","Pneumonia","Covid 19","Kindey stone","Brain Tumor","Heart"])
if a=="Fever":
    st.title("Fever")
    st.write("Please check your body temparature")
    f=st.number_input("Enter your body temperature in fahrenheit")
    if f is not None:
        if(f>98.6):
            st.write("You are suffering with fever. Please take care about your health.")
            st.header("Medicines for Fever")
            st.markdown("""
            - Acetaminophen (Tylenol)
            - Ibuprofen (Advil, Motrin)
            - Aspirin
            - Naproxen (Aleve)
            """)
        else:
            st.write("Your health is good.")
    st.markdown("---")
    st.write("Fever is a medical condition characterized by an elevated body temperature above the normal range, which is typically 98.6 degrees Fahrenheit (37 degrees Celsius). Fever is a natural response of the body to infection, inflammation, or other medical conditions.")
    st.subheader("Symptoms of Fever")
    st.markdown("""
    - flu
    - common cold
    - pneumonia
    - urinary tract infections
    - strep throat
    """)

elif a=="Sugar":
    st.title("Blood Sugar Level Calculator")
    st.write("Diabetes is a serious health condition. For its treatment to work, you must learn to manage blood sugar levels. You can use a blood sugar level calculator to understand your current condition and practice self-management better.")
    with st.form("sugar"):
        fbs=st.text_input("Enter Your Fast Blood Sugar (FBS) Reading :")
        pbs=st.text_input("Enter Your Post Blood Sugar (PBS) Reading :")
        sugar=st.form_submit_button("Submit")
        if sugar is not None:
            if((int(fbs)>=70 & int(fbs)<=100) and (int(pbs)<=140) ):
                st.write("You doesn't have sugar")
            elif(int(fbs)<=70 or int(pbs)<=140):
                st.write("You have low sugar")
                st.write("Madicine:")
                st.markdown("""
                - Increase Fiber Rich Foods
                - Ditch the Carbs
                - Avoid a Sugar Shock
                - Go Smart with Proteins   
                """)
            elif(int(fbs)>=100 or int(pbs)>=140):
                st.write("You have high sugar")
                st.write("Madicine:")
                st.markdown("""
                - Increase Fiber Rich Foods
                - Ditch the Carbs
                - Avoid a Sugar Shock
                - Go Smart with Proteins   
                """)              
            else:
                st.write("You have sugar")
                st.write("Madicine:")
                st.markdown("""
                - Increase Fiber Rich Foods
                - Ditch the Carbs
                - Avoid a Sugar Shock
                - Go Smart with Proteins   
                """)

elif a=="Details":
    st.title("Patient Details")
    st.write("Please Enter the details properly")
    with st.form("form2"):
        a,b=st.columns(2)
        a.text_input("Patient First Name")
        b.text_input("Patient Last Name")
        age1,gen=st.columns(2)
        age1.slider("Age",1,100)
        gen.selectbox("Gender",["Male", "Female","Other"])

        b1,b2,b3=st.columns(3)
        b1.number_input("Weight")
        b2.number_input("Height")
        b3.selectbox("Choose your blood group",["A+","A-","B+","B-","O+","O-","AB+","AB-"])
        st.text_area("Address")
        st.text_input("Phone Number")
        st.form_submit_button("Save")
    
    
elif a=="Malaria":
    st.title("Malaria")
    st.write("Malaria is a life-threatening disease caused by the Plasmodium parasite, which is transmitted to humans through the bites of infected female Anopheles mosquitoes. Malaria is a major public health problem, particularly in tropical and subtropical regions of the world.")
    st.subheader("Symptoms of Malaria")
    st.markdown("""
    - fever
    - chills
    - headache
    - muscle aches
    - fatigue
    - can lead to complications such as anemia, kidney failure, seizures, and coma.
    """)
    model = tf.keras.models.load_model('malaria_detect.h5')

    class_labels = ["Parasitised","Uninfected"]

    st.set_option('deprecation.showfileUploaderEncoding', False)
    img5 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if img5 is not None:
        file_bytes = np.asarray(bytearray(img5.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (150, 150))
        img = img.astype('float32') / 255.
        img = img.reshape(1, 150, 150, 3)
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Predicted class label:", predicted_class_label)
        st.header("Medicines for Malaria")
        st.markdown("""
        - Chloroquine
        - Artemisinin-based combination therapies (ACTs)
        - Quinine
        """)
elif a=="Pneumonia":
    st.title("Pneumonia")
    st.write("Pneumonia is a respiratory infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. It can be caused by bacteria, viruses, or fungi, and it can range in severity from mild to life-threatening.")
    st.subheader("Symptoms of Pneumonia")
    st.markdown("""
    - cough
    - fever
    - chest pain
    - shortness of breath
    - fatigue
    - Other symptoms may include chills, sweating, headache, muscle pain, and loss of appetite.
    """)
    model = tf.keras.models.load_model('pneumonia.h5')
    class_labels = ["Viral", "Bacterial", "Normal"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    img4 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if img4 is not None:
        file_bytes = np.asarray(bytearray(img4.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        img = cv2.resize(img, (224, 224))

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Predicted class label:", predicted_class_label)
        st.header("Medicines  for Pneumonia")
        st.markdown("""
        - Antibiotics like penicillin
        - Antiviral  like oseltamivi
        - bronchodilators like albuterol
        - Corticosteroids
        """)
    else:
        st.write("Please upload the image properly")
elif a=="Covid 19":
    st.title("Covid 19")
    st.write("COVID-19 is a highly contagious respiratory illness caused by the SARS-CoV-2 virus. It was first identified in Wuhan, China in December 2019 and has since become a global pandemic, affecting millions of people worldwide.")
    st.subheader("Symptoms of Covid 19")
    st.markdown("""
    - severe and may include fever, 
    - cough
    - shortness of breath
    - fatigue
    - loss of taste or smell
    - sore throat
    - Some people may experience no symptoms at all or only mild symptoms, 
    
    """)
    model = load_model('covid.h5')
    class_labels = ['COVID-19', 'Normal']

    img3=st.file_uploader("select a image")
    def predict(image):
        img = image.resize((224,224))
        x = np.array(img).astype('float32')/255
        x = np.expand_dims(x, axis=0)
        x /= 255.

        preds = model.predict(x)[0]

        predicted_class = np.argmax(preds)
        predicted_label = class_labels[predicted_class]

        return predicted_label, preds[predicted_class]
    
    if img3 is not None:
        image = Image.open(img3)
        predicted_label, confidence = predict(image)
        st.image(image, caption=f'Predicted class: {predicted_label}, Confidence: {confidence:.2f}')
        st.header("Medicines for Covid 19")
        st.markdown("""
        - Remdesivir
        - Dexamethasone
        - Tocilizumab
        - Tocilizumab
        """)
    else:
        st.write("Please upload image properly")

    
elif a=="Kindey stone":
    st.title("Kindey stone")
    st.write("Kidney stones are hard, crystal-like deposits that form in the kidneys, the organs that filter waste products from the blood and produce urine. Kidney stones can range in size from a grain of sand to a golf ball, and they can cause intense pain and discomfort when they pass through the urinary tract.")
    st.subheader("Symptoms of Kindey stone")
    st.markdown("""
    - severe pain in the back side
    - nausea
    - vomiting
    - blood in the urine
    - Some people may also experience fever, chills, or difficulty urinating.
    """)
    
    model = load_model('kidney.h5')

    class_labels = ["cyst","normal","stone","tumor"]

    img1=st.file_uploader("select a image")
    def predict(image):
        img = image.resize((224,224))
        x = np.array(img).astype('float32')/255
        x = np.expand_dims(x, axis=0)
        x /= 255
        preds = model.predict(x)[0]
        predicted_class = np.argmax(preds)
        predicted_label = class_labels[predicted_class]
        return predicted_label, preds[predicted_class]
    
    if img1 is not None:
        image = Image.open(img1)
        predicted_label, confidence = predict(image)
        st.image(image, caption=f'Predicted class: {predicted_label}, Confidence: {confidence:.2f}')
        st.header("Medicines for Kindey Stone")
        st.markdown("""
        - Pain relievers
        - Alpha-blockers 
        - Calcium channel blockers
        - Antibiotics
        """)
    else:
        st.write("Please upload image properly")

    
elif a=="Brain Tumor":
    st.title("Brain Tumor")
    st.write("A brain tumor is an abnormal growth of cells in the brain or surrounding tissues that can be either benign (non-cancerous) or malignant (cancerous). Brain tumors can develop in people of any age, but they are most commonly diagnosed in adults.")
    st.subheader("Symptoms of Brain Tumor ")
    st.markdown("""
    - headaches
    - seizures
    - nausea
    - vomiting
    - vision or hearing changes
    - memory or speech problems
    - weakness or numbness in the arms or legs""")
    model = load_model('brain-tumor-model.h5')
    class_labels = ['Pituitary', 'No Tumor','Meningioma','Glioma']

    img=st.file_uploader("select a image")
    def predict(image):
        img = image.resize((150, 150))
        x = np.array(img).astype('float32')/255
        x = np.expand_dims(x, axis=0)
        x /= 255.

        preds = model.predict(x)[0]

        predicted_class = np.argmax(preds)
        predicted_label = class_labels[predicted_class]

        return predicted_label, preds[predicted_class]
    
    if img is not None:
        image = Image.open(img)
        predicted_label, confidence = predict(image)
        st.image(image, caption=f'Predicted class: {predicted_label}, Confidence: {confidence:.2f}')
        st.header("Medicines for Brain Tumor")
        st.markdown("""
        - Chemotherapy
        - Targeted therapy
        - Corticosteroids
        - Anti-seizure medication
        """)
    else:
        st.write("Please upload image properly")

elif a=="Heart":
    st.title("Heart Disease Predication")
    st.write("Heart attack prediction is one of the serious causes of morbidity in the world’s population. The clinical data analysis includes a very crucial disease i.e., cardiovascular disease as one of the most important sections for the prediction. Data Science and machine learning (ML) can be very helpful in the prediction of heart attacks in which different risk factors like high blood pressure, high cholesterol, abnormal pulse rate, diabetes, etc... can be considered.")
    st.subheader("Symptoms of Heart Disease ")
    st.markdown("""
    - Chest pain
    - chest tightness
    - chest pressure
    - chest discomfort (angina)
    - Shortness of breath.
    - Pain in the neck, jaw, throat, upper belly area or back.
    """)
    heart = load_model('heart_disease.h5')
    with st.form("form1",clear_on_submit=True):
        a,b =st.columns(2)
        a.text_input("First Name")
        b.text_input("Last Name")
        a1,a2,a3=st.columns(3)
        age=a1.slider("age")
        print(age)
        s1=a2.selectbox("sex",["male","female"])
        if s1 is not None:
            if s1== "male":
                sex=1
            elif s1 == "female":
                sex=0
            
        c=a3.selectbox("Chest pain type",options=["Typical angina","atypical angina","non-anginal pain","asymptomatic"])
        if c is not None:
            if c== "Typical angina":
                cpt=1
            elif c == "atypical angina":
                cpt=2
            elif c == "non-anginal pain":
                cpt=3
            elif c == "asymptomatic":
                cpt=4
            
        a4=st.number_input("Resting blood pressure (in mm Hg on admission to the hospital)")
        a5=st.number_input("Enter serum cholestoral in mg/dl")
        a6=st.selectbox("Choose if fasting blood sugar > 120 mg/dl",["True","False"])
        if a6=="True":
            fbs=1
        elif a6=="False":
            fbs=0

        a7=st.selectbox("Select the resting electrocardiographic results",["normal","having ST-T wave abnormality","showing probable or definite left ventricular hypertrophy"])
        if a7=="normal":
            restecg=0
        elif a7=="having ST-T wave abnormality":
            restecg=1
        elif a7=="showing probable or definite left ventricular hypertrophy by Estes' criteria":
            restecg=2

        a8=st.number_input("Enter the maximum heart rate achieved")
        a9=st.selectbox("Select the exercise induced angina",["Yes","No"])
        if a9=="Yes":
            exang=1
        elif a9=="No":
            exang=0
        a10=st.number_input("Enter the ST depression induced by exercise relative to rest")
        a11=st.selectbox("Select the slope of the peak exercise ST segment:",["upsloping","flat","downsloping"])
        if a11=="upsloping":
            slope=1
        elif a11=="flat":
            slope=2
        elif a11=="downsloping":
            slope=3

        a12=st.number_input("number of major vessels (0-3) colored by flourosopy")
        a13=st.selectbox("Select the defect",["normal","fixed defect","reversable defect"])
        #3 = normal; 6 = fixed defect; 7 = reversable defect"
        if a13 == "normal":
            thal=3
        elif a13== "fixed defect":
            thal=6
        elif a13 == "reversable defect":
            thal=7
        submitted = st.form_submit_button("Submit")

        arr=[age,sex,cpt,a4,a5,fbs,restecg,a8,exang,a10,slope,a12,thal]
        if submitted is not None:
            input_data = np.array([arr])
            input_data = (input_data - np.mean(input_data, axis=0)) / np.std(input_data, axis=0)
            predictions = heart.predict(input_data)
            if predictions > 0.5:
                st.success("Patient is likely to have heart disease.")
                st.header("Medicines for Heart Disease")
                st.markdown("""
                - Amlodipine (Norvasc)
                - Diltiazem (Cardizem, Tiazac)
                - Felodipine (Plendil)
                - Nifedipine (Adalat, Procardia)
                - Nimodipine (Nimotop)
                - Nisoldipine (Sular)
                - Verapamil (Calan, Verelan)
                """)
            else:
                st.error("Patient is unlikely to have heart disease.")
                st.header("Medicines for Heart Disease")
                st.markdown("""
                - Amlodipine (Norvasc)
                - Diltiazem (Cardizem, Tiazac)
                - Felodipine (Plendil)
                - Nifedipine (Adalat, Procardia)
                - Nimodipine (Nimotop)
                - Nisoldipine (Sular)
                - Verapamil (Calan, Verelan)
                """)
        else:
            st.write("please enter your details properly")

st.subheader("Team Members")
a,b,c,d,e=st.columns(5)
a.image("avi.jpg",caption="JINKA THE AVIRAJ")
b.image("gautham.jpg",caption="GOUTHAM SANKAR S")
c.image("vamsi.jpg",caption="KOTA RAMASAI VAMSI")
d.image("rajkumar.jpg",caption="BALA VENTAKA RAJKUMAR T S")
e.image("MAHI.jpg",caption="DEVISETTY MAHIDHAR")
