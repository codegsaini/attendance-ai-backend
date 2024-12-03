import face_recognition
import cv2
import pickle


def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print("Model loaded successfully.")
        return known_face_encodings, known_face_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return [], []


import face_recognition
import cv2


def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image.")
        return

    print("Image loaded successfully.")

    # Convert the image to RGB (OpenCV loads it in BGR)
    rgb_image = image[:, :, ::-1]

    # Detect face locations using HOG model or CNN model
    face_locations = face_recognition.face_locations(rgb_image, model="cnn")  # Using CNN model for better accuracy
    print(f"Found face locations: {face_locations}")

    # Draw bounding boxes around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle around faces

    # Show the image with detected faces
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test with your image
test_image_path = "dataset/pce21cy017/1 (4).jpg"  # Replace with a valid image path
detect_faces_in_image(test_image_path)


# # Run the face recognition on a test image
# def main():
#     model_path = "face_recognition_model.pkl"  # Path to your saved model
#     known_face_encodings, known_face_names = load_model(model_path)
#
#     # Path to a test image
#     test_image_path = "dataset/pce21cy017/1 (4).jpg"  # Replace with a valid image path
#     detect_faces_in_image(test_image_path)

#
# if __name__ == "__main__":
#     main()
