import cv2
import json
import face_recognition
import time
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os
import csv
from datetime import datetime

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize the image to a consistent size (e.g., 100x100)
            img = cv2.resize(img, (100, 100))
            images.append(img)
            labels.append(int(filename.split('_')[0]))
    return images, labels

def train_face_recognition_model(images, labels):
    images_np = np.array(images)
    labels_np = np.array(labels)

    images_flatten = images_np.reshape((images_np.shape[0], -1))
    pca = PCA(n_components=3) 
    images_pca = pca.fit_transform(images_flatten)
    svc = SVC()
    svc.fit(images_pca, labels_np)

    return pca, svc
print('Training complete')
def recognize_faces(image, pca, svc):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_flatten = gray.reshape(1, -1)
    img_pca = pca.transform(img_flatten)
    label = svc.predict(img_pca)[0]

    return label

def detect_faces(image):
    # Check if the frame is not empty
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def write_attendance_to_csv(filename, label):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date_time, label])


def create_face_encodings_for_person_and_save(person_folder_path, output_json_path):
    person_face_encodings = []

    image_files = [f for f in os.listdir(person_folder_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

    if not image_files:
        print(f"No image files found in '{person_folder_path}'.")
        return None

    for image_file in image_files:
        image_path = os.path.join(person_folder_path, image_file)

        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
            face_encoding_list = face_encoding.tolist()
            person_face_encodings.append(face_encoding_list)

    person_name = os.path.basename(person_folder_path)

    with open(output_json_path, 'w') as json_file:
        json.dump({person_name: person_face_encodings}, json_file)

person_folder_path = "Swaksh"
output_json_path = f"{os.path.basename(person_folder_path)}_face_encodings.json"
create_face_encodings_for_person_and_save(person_folder_path, output_json_path)


def identify_faces_and_mark_attendance(json_file_path, csv_file_path, threshold=0.5, unlock_duration=3):
    # Load face encodings from the JSON file
    try:
        with open(json_file_path, 'r') as json_file:
            face_encodings_dict = json.load(json_file)
    except FileNotFoundError:
        print("Error: Face encodings JSON file not found.")
        return

    # Open the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Variables for tracking the time the face has been consistently detected
    unlock_start_time = None
    unlock_detected = False

    # Create or open the CSV file for attendance
    with open(csv_file_path, mode='a', newline='') as csv_file:
        fieldnames = ['Date-Time', 'Person']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Check if the file is empty, write the header if needed
        if csv_file.tell() == 0:
            writer.writeheader()

        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Find face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)

            # Reset variables for each frame
            unlock_detected_this_frame = False

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check if any of the stored encodings match the input face
                match_found = False
                for name, encodings in face_encodings_dict.items():
                    for stored_encoding in encodings:
                        match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=threshold)
                        if match[0]:
                            # Draw a bounding box around the face and display the name
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            unlock_detected_this_frame = True
                            match_found = True
                            break

                if not match_found:
                    # Draw a bounding box around the face with red color for unmatched face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    unlock_start_time = None  # Reset the timer

            # Update the time variables
            if unlock_detected_this_frame:
                if unlock_start_time is None:
                    unlock_start_time = time.time()
                elif time.time() - unlock_start_time >= unlock_duration:
                    unlock_detected = True

                    # Mark attendance in CSV file
                    now = datetime.now()
                    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    person_name = person_folder_path  # Replace with the actual person's name
                    writer.writerow({'Date-Time': date_time, 'Person': person_name}, 'Status': 'Present'})
                    
                    # Release the camera and close the window
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Exit the loop

            # Display the frame
            cv2.imshow('Face Identification', frame)

            # Check for exit key 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera
        cap.release()
        cv2.destroyAllWindows()

# Example usage
json_file_path = f"{os.path.basename(person_folder_path)}_face_encodings.json"
csv_file_path = "attendance.csv"
identify_faces_and_mark_attendance(json_file_path, csv_file_path)