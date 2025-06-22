from camera_module import Camera
import cv2
import sys

def main():
    try:
        # Initialize camera (try different indices if needed)
        camera = Camera(camera_index=0)
        
        print("Press 'q' to exit...")
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("Error: Failed to capture frame")
                break
            
            # Display FPS on frame
            fps_text = f"FPS: {camera.fps:.1f}"
            cv2.putText(
                img=frame,
                text=fps_text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2
            )
            
            cv2.imshow("Camera Test", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    main()