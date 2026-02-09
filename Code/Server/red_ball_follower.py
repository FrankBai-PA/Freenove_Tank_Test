# Red Ball Follower - Tank robot follows a red ball using camera vision
# The robot will stop when it cannot see the red ball

import time
import numpy as np
import cv2
from camera import Camera
from car import Car

class RedBallFollower:
    def __init__(self):
        """Initialize the red ball follower with camera and car components."""
        print("Initializing Red Ball Follower...")
        self.camera = Camera(stream_size=(640, 480))  # Initialize camera with stream size
        self.car = Car()  # Initialize car (motors)
        
        # Start camera streaming
        print("Starting camera stream...")
        self.camera.start_stream()
        time.sleep(1)  # Give camera time to initialize
        
        # Red color range in HSV (adjust these values based on your red ball)
        # Lower and upper bounds for red color detection
        # Red wraps around 0 in HSV, so we need two ranges
        self.lower_red1 = np.array([0, 100, 100])    # Lower bound for red (0-10)
        self.upper_red1 = np.array([10, 255, 255])  # Upper bound for red (0-10)
        self.lower_red2 = np.array([170, 100, 100]) # Lower bound for red (170-180)
        self.upper_red2 = np.array([180, 255, 255]) # Upper bound for red (170-180)
        
        # Minimum area to consider as a valid ball detection
        self.min_ball_area = 500  # Adjust based on camera distance and ball size
        
        # Control parameters
        self.center_tolerance = 50  # Pixels from center to consider "centered"
        self.base_speed = 1500  # Base forward speed
        self.turn_speed = 2000  # Turn speed when ball is off-center

        # Display window (set to False for headless runs)
        self.show_window = True
        
        print("Red Ball Follower initialized. Press Ctrl+C to stop.")
    
    def detect_red_ball(self, img):
        """
        Detect red ball in the image (BGR).
        Returns: (found, center_x, center_y, area) tuple
        """
        try:
            if img is None:
                return False, None, None, 0
            
            # Convert BGR to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create mask for red color (handle red wrapping around 0)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, None, None, 0
            
            # Find the largest contour (assumed to be the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if area is large enough
            if area < self.min_ball_area:
                return False, None, None, 0
            
            # Calculate center of the ball
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                return True, center_x, center_y, area
            
            return False, None, None, 0
            
        except Exception as e:
            print(f"Error in detect_red_ball: {e}")
            return False, None, None, 0
    
    def follow_ball(self):
        """Main loop to follow the red ball."""
        frame_width = 640  # Camera stream width
        center_x_target = frame_width // 2  # Center of the frame
        
        try:
            while True:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is None:
                    print("No frame received, stopping...")
                    self.car.motor.setMotorModel(0, 0)
                    time.sleep(0.1)
                    continue

                # Decode JPEG bytes into an image for processing/display
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    print("Failed to decode frame, stopping...")
                    self.car.motor.setMotorModel(0, 0)
                    time.sleep(0.1)
                    continue

                # Detect red ball
                found, ball_x, ball_y, area = self.detect_red_ball(img)
                
                if not found:
                    # Ball not found - stop the robot
                    print("Red ball not detected - stopping...")
                    self.car.motor.setMotorModel(0, 0)
                # Display the current frame with overlays (if enabled)
                if self.show_window:
                    display_img = img.copy()
                    height, width = display_img.shape[:2]
                    cv2.line(display_img, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

                    if found:
                        cv2.circle(display_img, (ball_x, ball_y), 12, (0, 255, 0), 2)
                        cv2.putText(display_img, f"area={area:.0f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_img, "No red ball", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Red Ball Follower", display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

                if not found:
                    time.sleep(0.1)
                    continue
                
                # Calculate offset from center
                offset = ball_x - center_x_target
                
                # Control logic based on ball position
                if abs(offset) < self.center_tolerance:
                    # Ball is centered - move forward
                    print(f"Ball centered (area: {area:.0f}) - moving forward")
                    self.car.motor.setMotorModel(self.base_speed, self.base_speed)
                elif offset < 0:
                    # Ball is on the left - turn left
                    print(f"Ball on left (offset: {offset}, area: {area:.0f}) - turning left")
                    self.car.motor.setMotorModel(-self.turn_speed, self.turn_speed)
                else:
                    # Ball is on the right - turn right
                    print(f"Ball on right (offset: {offset}, area: {area:.0f}) - turning right")
                    self.car.motor.setMotorModel(self.turn_speed, -self.turn_speed)
                
                time.sleep(0.05)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\nStopping Red Ball Follower...")
            self.car.motor.setMotorModel(0, 0)
            self.cleanup()
        except Exception as e:
            print(f"Error in follow_ball: {e}")
            self.car.motor.setMotorModel(0, 0)
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.camera.stop_stream()
        self.camera.close()
        self.car.close()
        if self.show_window:
            cv2.destroyAllWindows()
        print("Cleanup complete.")

def main():
    """Main entry point."""
    follower = RedBallFollower()
    follower.follow_ball()

if __name__ == '__main__':
    main()
