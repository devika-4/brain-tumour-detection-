import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import xgboost as xgb

# Load the trained XGBoost model
xgb_model = joblib.load("brain_tumor_xgboost_model.pkl")

# Load Pretrained VGG16 Model for Feature Extraction
def load_vgg16_model():
    vgg_model = VGG16(weights='imagenet', include_top=False)
    return Model(inputs=vgg_model.input, outputs=vgg_model.output)

model = load_vgg16_model()

# Debugging function for feature selection (Ensure it's same as training)
def select_important_features(features):
    rf_model = joblib.load("random_forest_model.pkl")  # Load the correct file
    print("Extracted Features Shape:", features.shape)
    
    try:
        selected_features = rf_model.transform(features)  # Apply feature selection
        print("Selected Features Shape:", selected_features.shape)
        return selected_features
    except AttributeError:
        print("Error: RandomForestClassifier does not have a transform() function.")
        return features  # Skip feature selection and use all features

# Function to Predict Tumor Type
def predict_tumor(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    
    print("Preprocessed Image Shape:", img_array.shape)
    print("Preprocessed Image Values (First 5 pixels):", img_array.flatten()[:5])

    features = model.predict(img_array)  # Extract features using VGG16
    features_flattened = features.reshape(1, -1)  # Flatten features
    
    selected_features = select_important_features(features_flattened)  # Apply feature selection
    
    # Check probabilities
    probs = xgb_model.predict_proba(selected_features)
    print("Prediction Probabilities:", probs)

    # Get final prediction
    class_labels = {0: "Glioma", 1: "Meningioma", 2: "Pituitary", 3: "No Tumor"}
    predicted_class = class_labels[np.argmax(probs)]

    print(f"Predicted Tumor Type: {predicted_class}")

# Provide the new image path
image_path = r"C:\Users\layas\OneDrive\Desktop\BrainTumorProject\BrainTumorDataset\NewTestImages\00R.jpg"
predict_tumor(image_path)





