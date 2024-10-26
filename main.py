import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import pyttsx3
from plyer import notification, vibrator
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

# Load the face detector and the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fatigue detection configuration
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
counter = 0

# Function to calculate the Euclidean distance
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

class FatigueDetectionApp(App):
    def build(self):
        # Set up the layout
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        
        # Label for displaying messages (e.g., "Drowsiness Detected")
        self.label = Label(text="Fatigue Detection System", size_hint=(1, 0.1))
        layout.add_widget(self.label)

        # Start the OpenCV camera
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        return layout

    def update(self, dt):
        global counter
        
        # Read the next frame
        ret, frame = self.capture.read()
        if not ret:
            return

        # Convert the frame to grayscale for dlib processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Process each detected face
        for face in faces:
            shape = predictor(gray, face)
            shape_np = np.zeros((68, 2), dtype="int")

            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            # Calculate the EAR for both eyes
            leftEye = shape_np[36:42]
            rightEye = shape_np[42:48]
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Check if EAR is below the threshold
            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.label.text = "Drowsiness Detected!"
                    pyttsx3.speak("Wake up!")
            else:
                counter = 0
                self.label.text = "Fatigue Detection System"

            # Draw eye contours for visualization
            cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)

        # Convert the frame to texture for displaying in Kivy
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def on_stop(self):
        # Release the camera when the app is closed
        self.capture.release()

if __name__ == '__main__':
    FatigueDetectionApp().run()


