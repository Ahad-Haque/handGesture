import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import sys

class HandGestureRecognizer:
    def __init__(self, camera_index=0):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # FPS calculation
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Camera index
        self.camera_index = camera_index
    
    def init_camera(self):
        """Initialize camera with error handling"""
        print(f"Initializing camera {self.camera_index}...")
        
        # Try different backends in addition to different camera indices
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),  # Windows preferred
            (cv2.CAP_ANY, "Default"),       # Platform default
            (cv2.CAP_MSMF, "Media Foundation"),  # Alternative Windows backend
        ]
        
        for idx in [self.camera_index, 0, 1, 2]:
            for backend, backend_name in backends:
                print(f"Trying camera {idx} with {backend_name} backend...")
                try:
                    # Use the specific backend API
                    cap = cv2.VideoCapture(idx, backend)
                    
                    # Wait a moment for camera initialization
                    time.sleep(1)
                    
                    if cap.isOpened():
                        print(f"Camera {idx} opened successfully with {backend_name}!")
                        
                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        
                        # Test read multiple frames
                        for _ in range(5):  # Try reading a few frames
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                print(f"Failed to read frame from camera {idx}")
                                cap.release()
                                break
                            # Short delay between test frames
                            time.sleep(0.1)
                        else:  # This executes if the for loop completes normally
                            print(f"Camera {idx} with {backend_name} is working properly!")
                            return cap
                        
                        # If we got here, frame reading failed
                        print(f"Camera {idx} opened but cannot read frames")
                        cap.release()
                    else:
                        print(f"Failed to open camera {idx} with {backend_name}")
                        if cap is not None:
                            cap.release()
                except Exception as e:
                    print(f"Error with camera {idx} and {backend_name}: {e}")
        
        # If no camera available, try listing available cameras
        self.list_available_cameras()
        return None

    def list_available_cameras(self):
        """List available cameras using platform-specific methods"""
        print("\nAttempting to list available cameras...")
        
        # Platform specific camera listing
        if sys.platform.startswith('win'):
            # Windows - try to enumerate cameras
            try:
                import wmi
                wmi_obj = wmi.WMI()
                wmi_cameras = wmi_obj.Win32_PnPEntity(ConfigManagerErrorCode=0)
                
                print("\nDetected camera devices on Windows:")
                camera_found = False
                for camera in wmi_cameras:
                    if "camera" in camera.Caption.lower() or "webcam" in camera.Caption.lower():
                        print(f" - {camera.Caption}")
                        camera_found = True
                
                if not camera_found:
                    print("No camera devices detected via WMI")
            except Exception as e:
                print(f"Could not list Windows cameras: {e}")
        elif sys.platform.startswith('linux'):
            # Linux - check /dev/video*
            try:
                import glob
                cameras = glob.glob('/dev/video*')
                print("\nDetected video devices on Linux:")
                for camera in cameras:
                    print(f" - {camera}")
            except Exception as e:
                print(f"Could not list Linux cameras: {e}")
        
        print("\nTroubleshooting steps:")
        print("1. Ensure your camera is properly connected")
        print("2. Close any applications that might be using the camera (including web browsers)")
        print("3. Check device manager to verify camera is properly installed")
        print("4. Try updating camera drivers")
        print("5. Try a different USB port")
        print("6. Restart your computer")

    
    def classify_gesture(self, landmarks):
        """Classify hand gesture based on landmarks"""
        if not landmarks:
            return "None", 0.0
        
        # Extract landmark coordinates
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        points = np.array(points)
        
        # Simple rule-based gesture classification
        gesture, confidence = self.analyze_finger_positions(points)
        return gesture, confidence
    
    def analyze_finger_positions(self, points):
        """Analyze finger positions to determine gesture"""
        # Landmark indices for fingertips and bases
        THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
        INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
        MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
        RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
        PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17
        WRIST = 0
        
        # Check if fingers are extended
        fingers_up = []
        
        # Thumb (special case - check x coordinate)
        if points[THUMB_TIP][0] > points[THUMB_IP][0]:  # Right hand
            fingers_up.append(points[THUMB_TIP][0] > points[THUMB_MCP][0])
        else:  # Left hand
            fingers_up.append(points[THUMB_TIP][0] < points[THUMB_MCP][0])
        
        # Other fingers (check y coordinate)
        for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), 
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
            fingers_up.append(points[tip][1] < points[pip][1])
        
        # Count extended fingers
        extended_count = sum(fingers_up)
        
        # Gesture recognition logic
        if extended_count == 0:
            return "Fist", 0.9
        elif extended_count == 1:
            if fingers_up[1]:  # Index finger
                return "Pointing", 0.9
            elif fingers_up[0]:  # Thumb
                return "Thumbs Up", 0.9
        elif extended_count == 2:
            if fingers_up[1] and fingers_up[2]:  # Index and middle
                return "Peace", 0.9
            elif fingers_up[1] and fingers_up[4]:  # Index and pinky
                return "Rock", 0.9
        elif extended_count == 3:
            if fingers_up[0] and fingers_up[1] and fingers_up[4]:
                return "Rock", 0.85
            elif fingers_up[1] and fingers_up[2] and fingers_up[3]:
                return "Three", 0.9
        elif extended_count == 4:
            if not fingers_up[0]:
                return "Four", 0.9
        elif extended_count == 5:
            return "Stop", 0.95
        
        # Check for OK gesture
        thumb_index_dist = np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP])
        if thumb_index_dist < 0.05 and fingers_up[2] and fingers_up[3] and fingers_up[4]:
            return "OK", 0.9
        
        return "Unknown", 0.3
    
    def calculate_fps(self):
        """Calculate frames per second"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_queue.append(fps)
        self.last_time = current_time
        return np.mean(self.fps_queue)
    
    def draw_info(self, image, fps):
        """Draw info on image"""
        # Draw FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw help text
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def run(self):
        """Main loop for hand gesture recognition"""
        # Initialize camera
        cap = self.init_camera()
        if cap is None:
            print("ERROR: Could not initialize any camera!")
            print("Please check:")
            print("1. Your camera is connected")
            print("2. No other application is using the camera")
            print("3. You have proper camera drivers installed")
            return
        
        # Create window
        window_name = 'Hand Gesture Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Create a blank frame to show initially
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Initializing camera...", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, blank_frame)
        cv2.waitKey(1)
        
        print("Starting main loop...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"Failed to read frame at count {frame_count}")
                
                # Show error message
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error - No Frame", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, error_frame)
                
                # Try to reinitialize camera
                if frame_count > 10:  # Give it some attempts
                    print("Attempting to reinitialize camera...")
                    cap.release()
                    cap = self.init_camera()
                    if cap is None:
                        break
                    frame_count = 0
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                continue
            
            # Successfully got a frame
            frame_count = 0
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            # Draw FPS
            fps = self.calculate_fps()
            frame = self.draw_info(frame, fps)
            
            # Draw hand landmarks and classify gestures
            if results.multi_hand_landmarks:
                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):
                    
                    # Draw hand skeleton
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Classify gesture
                    gesture, confidence = self.classify_gesture(hand_landmarks)
                    
                    # Get hand label (Left/Right)
                    hand_label = handedness.classification[0].label
                    
                    # Draw bounding box and label
                    h, w, _ = frame.shape
                    landmarks_array = np.array([[l.x * w, l.y * h] for l in hand_landmarks.landmark])
                    x_min, y_min = np.min(landmarks_array, axis=0).astype(int)
                    x_max, y_max = np.max(landmarks_array, axis=0).astype(int)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Draw box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{hand_label}: {gesture}"
                    cv2.rectangle(frame, (x_min, y_min - 25), 
                                (x_min + 150, y_min), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (x_min + 5, y_min - 7),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No hands detected
                cv2.putText(frame, "Show your hand", (250, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exiting...")
                break
        
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Done!")

if __name__ == "__main__":
    print("Starting Hand Gesture Recognition System...")
    print("Make sure your camera is connected and not being used by another application.")
    print()
    
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        