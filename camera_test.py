# camera_test.py
import cv2
import numpy as np

def test_cameras():
    """Test all available cameras"""
    print("Testing available cameras...")
    
    for i in range(5):  # Test first 5 camera indices
        print(f"\nTesting camera {i}...")
        
        # Try different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_ANY, "Any available")
        ]
        
        for backend, name in backends:
            print(f"  Trying {name} backend...")
            cap = cv2.VideoCapture(i, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"    ✓ Camera {i} works with {name}!")
                    print(f"    Frame shape: {frame.shape}")
                    
                    # Show test frame
                    cv2.imshow(f'Camera {i} - {name}', frame)
                    cv2.waitKey(1000)  # Show for 1 second
                    cv2.destroyAllWindows()
                    
                    cap.release()
                    return i, backend
                else:
                    print(f"    × Camera {i} opened but can't read frames")
            else:
                print(f"    × Camera {i} failed to open")
            
            cap.release()
    
    return None, None

if __name__ == "__main__":
    print("Camera Detection Test")
    print("====================")
    
    # Check if OpenCV is properly installed
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test cameras
    camera_idx, backend = test_cameras()
    
    if camera_idx is not None:
        print(f"\nWorking camera found: Index {camera_idx} with backend {backend}")
        print("\nTesting continuous capture...")
        
        cap = cv2.VideoCapture(camera_idx, backend)
        cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while frame_count < 100:  # Test 100 frames
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
            else:
                print(f"Failed to read frame {frame_count}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCaptured {frame_count} frames successfully")
    else:
        print("\nNo working camera found!")
        print("\nTroubleshooting steps:")
        print("1. Check if your camera is connected")
        print("2. Close any applications using the camera")
        print("3. Check Windows Camera app to see if camera works")
        print("4. Update your camera drivers")
        print("5. Try running as administrator")