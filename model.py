import cv2
import face_recognition
import numpy as np
import os
import pickle


# Step 1: Load images of classmates and encode faces
def load_images_from_folder(folder_path):
    """
    Load images from the folder and return the list of images and names.
    The folder should contain subfolders, where each subfolder is named after a person.
    Each subfolder should contain multiple images of that person.
    """
    images = []
    names = []
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):  # Only process folders (representing each person)
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    names.append(person_name)  # Use folder name as the person's name
    return images, names


def encode_faces(images):
    """
    Encode the faces of all images provided and return their encodings.
    """
    encodings = []
    for image in images:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Get the face encodings in the image
        face_encoding = face_recognition.face_encodings(rgb_image)
        if face_encoding:  # Check if a face was found
            encodings.append(face_encoding[0])  # Assuming one face per image
    return encodings


def train_and_export_model(folder_path, export_path):
    # Load images and names from the folder
    images, names = load_images_from_folder(folder_path)

    # Encode faces of the known classmates
    known_face_encodings = encode_faces(images)

    # Save the encodings and names to a file (pickle format)
    with open(export_path, 'wb') as f:
        pickle.dump((known_face_encodings, names), f)
    print(f"Model saved to {export_path}")


# Provide the path to your dataset folder and export path
folder_path = "dataset"  # Replace with the path to your dataset
export_path = "face_recognition_model.pkl"  # Path where the model will be saved

train_and_export_model(folder_path, export_path)
