import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import joblib
from collections import Counter

# Load Pretrained VGG16 Model (without top layers)
def load_vgg16_model():
    vgg_model = VGG16(weights='imagenet', include_top=False)
    return Model(inputs=vgg_model.input, outputs=vgg_model.output)

# Define Image Size and Paths
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
TRAIN_PATH = r"C:\Users\layas\OneDrive\Desktop\BrainTumorProject\BrainTumorDataset\Training"
TEST_PATH = r"C:\Users\layas\OneDrive\Desktop\BrainTumorProject\BrainTumorDataset\Testing"

# Load Dataset Efficiently with Rescaling
def load_dataset(directory):
    dataset = image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    ).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = load_dataset(TRAIN_PATH)
test_dataset = load_dataset(TEST_PATH)

# Extract Features using CNN (VGG16)
def extract_features(dataset, model):
    feature_list, label_list = [], []
    for images, labels in dataset:
        images = preprocess_input(images)  # Ensure consistent preprocessing
        features = model.predict(images, verbose=1)
        feature_list.append(features.reshape(features.shape[0], -1))
        label_list.append(labels.numpy())
    return np.vstack(feature_list), np.hstack(label_list)

# Load the VGG16 model for feature extraction
model = load_vgg16_model()

# Extract Features for Training and Testing
X_train_feature, y_train = extract_features(train_dataset, model)
X_test_feature, y_test = extract_features(test_dataset, model)

# Check Class Distribution
print(f"Class Distribution in Training Data: {Counter(y_train)}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_feature, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Handle Class Imbalance (Use class_weight)
class_weights = {i: max(Counter(y_train).values()) / v for i, v in Counter(y_train).items()}
print(f"Class Weights: {class_weights}")

# Feature Selection using Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Use SelectFromModel to get the most important features
selector = SelectFromModel(rf_model, threshold="median", prefit=True)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test_feature)

# Apply XGBoost for Classification
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train_selected, y_train, eval_set=[(X_val_selected, y_val)], early_stopping_rounds=10, verbose=True)

# Evaluate on Validation Set
y_val_pred = xgb_model.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on Test Set
y_test_pred = xgb_model.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained XGBoost model
joblib.dump(xgb_model, "brain_tumor_xgboost_model.pkl")

# Save the trained Random Forest model (if needed)
joblib.dump(rf_model, "random_forest_model.pkl")

print("XGBoost model saved successfully as brain_tumor_xgboost_model.pkl")
