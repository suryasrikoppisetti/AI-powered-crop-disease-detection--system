import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it beautiful
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("✨ Image Classification")
st.markdown("### Upload an image and let AI do the magic!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )


def load_text_file(file_path):
    disease_info = {}
    with open(file_path, "r") as file:
        for line in file.readlines():
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                disease_info[parts[0].strip()] = parts[1].strip()
    return disease_info

symptoms_data = load_text_file("C:/Users/SURYA SRI/Downloads/SuryaKiran/SuryaKiran/symptoms.txt")
remedies_data = load_text_file("C:/Users/SURYA SRI/Downloads/SuryaKiran/SuryaKiran/remedies.txt")
# Main content
def load_model():
    # Load your trained model here
    
    model_path = "C:/Users/SURYA SRI/Downloads/SuryaKiran/SuryaKiran/crop_disease_model.keras"
    model = tf.keras.models.load_model(model_path)
    return model  # Your trained model
#Define Class Labels Manually
class_labels = ["potato_late_blight", "rice_LeafBLast", "tomato_target_spot"]

def process_image(image):
    # Add your image preprocessing logic here
    img = image.resize((150,150))  # Adjust size according to your model
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    

def main():
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image for classification"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("Analyzing image..."):
                # Load model
                model = load_model()
                
                # Process image
                processed_img = process_image(image)
                
                # Make prediction
                prediction = model.predict(processed_img)
                
                # Get predicted class
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction)
                
                # Display results
                st.success("Analysis Complete!")
                st.markdown(f'"### Results": {predicted_class} with {confidence:.2%} confidence')
                st.write("### Symptoms")
                # Get class labels from symptoms file
                if predicted_class in symptoms_data:
                    remedies = symptoms_data[predicted_class]
                else:
                    remedies = "No remedies found"
    
                st.write(remedies)
                
                # Add your prediction display logic here
                # Example:
                # for class_name, probability in zip(classes, prediction[0]):
                #     if probability > confidence_threshold:
                #         st.progress(float(probability))
                #         st.write(f"{class_name}: {probability:.2%}")

    else:
        st.markdown(
            """
            <div class="upload-text">
                👆 Upload an image to get started!
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
