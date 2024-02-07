import cv2
import json
import face_recognition
import time
from concurrent.futures import ThreadPoolExecutor
import threading

def identify_faces_from_camera(json_file_path, threshold=0.5, unlock_duration=3):
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

    # Skip every other frame for processing
    process_this_frame = True

    # Lock for synchronizing access to the OpenCV window
    lock = threading.Lock()

    # Function to perform face recognition in a separate thread
    def face_recognition_thread(frame):
        nonlocal unlock_detected, unlock_start_time

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
                        with lock:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        unlock_detected_this_frame = True
                        match_found = True
                        break

            if not match_found:
                # Draw a bounding box around the face with red color for unmatched face
                with lock:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                unlock_start_time = None  # Reset the timer

        # Update the time variables
        if unlock_detected_this_frame:
            if unlock_start_time is None:
                unlock_start_time = time.time()
            elif time.time() - unlock_start_time >= unlock_duration:
                unlock_detected = True
                print("Face unlocked. Phone is now accessible.")
        else:
            unlock_detected = False

    # Create a thread pool for face recognition
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Resize frame to half for smoother processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            if process_this_frame:
                # Submit the face recognition task to the thread pool
                executor.submit(face_recognition_thread, small_frame)

            process_this_frame = not process_this_frame

            # Display the frame
            with lock:
                cv2.imshow('Face Identification', frame)

            # Check for exit key 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Example usage
json_file_path = "face_encodings.json"
identify_faces_from_camera(json_file_path)
