import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained helmet detection model (Assuming you have a model named 'helmet_model')
# helmet_model = load_model('helmet_detection_model.h5')  # Example of loading the model

# Define a function to detect helmets
def detect_helmets(roi_color):
    # Placeholder function, you need to implement this according to your helmet detection model
    # Example: return helmet_boxes
    pass

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Filter out detections that are too high or too low to be considered as faces
    filtered_faces = []
    for (x, y, w, h) in faces:
        if y + h < frame.shape[0] * 0.85:  # Exclude detections too close to the top (likely glasses)
            filtered_faces.append((x, y, w, h))

    # Draw rectangle around the faces and detect eyes
    num_passengers = len(filtered_faces)
    for (x, y, w, h) in filtered_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        num_eyes = len(eyes)

        # Detect helmets
        helmet_boxes = detect_helmets(roi_color)
        if helmet_boxes:
            cv2.putText(frame, 'Helmet is Wearing', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Helmet is Not Wearing', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the number of passengers
    cv2.putText(frame, f'Passengers: {num_passengers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
