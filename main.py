import cv2
import numpy as np

# ==========================================
# CONFIGURATION: Choose your mode here
# ==========================================
USE_WEBCAM = False         # Set to True for live video, False for a static image file
IMAGE_PATH = r'C:\Users\saumy\opencv_project\shapes.png'  # If USE_WEBCAM is False, put your image path here


def process_frame(frame):
    # -----------------------------------------------------------------
    # REQUIREMENT 1: Image Preprocessing
    # -----------------------------------------------------------------
    # A. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # B. Apply Gaussian Blur to smooth out pixel noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # C. Apply Thresholding & Canny Edge Detection to extract outlines cleanly
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred, 50, 150)
    
    # -----------------------------------------------------------------
    # REQUIREMENT 2: Contour Detection & Noise Filtering
    # -----------------------------------------------------------------
    # Find outer object boundaries
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # REQUIREMENT 4a & 4b: Calculate Area and Perimeter
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Noise Filter: Ignore contours that are too small (e.g., less than 500 pixels in area)
        if area < 500:
            continue
            
        # -----------------------------------------------------------------
        # REQUIREMENT 3: Polygon Approximation & Classification
        # -----------------------------------------------------------------
        # Smooth out edges using Douglas-Peucker curve approximation
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        # REQUIREMENT 4c: Calculate Bounding Box
        x, y, w, h = cv2.boundingRect(approx)
        
        # Default name
        shape_name = f"Polygon ({vertices} sides)"
        
        # Classification Logic
        if vertices == 3:
            shape_name = "Triangle"
            
        elif vertices == 4:
            # Differentiate between Square and Rectangle using Aspect Ratio (width / height)
            aspect_ratio = float(w) / h
            # If width and height are within 5% of each other, it's a square
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
                
        elif vertices == 6:
            shape_name = "Hexagon"
            
        elif vertices > 6 and vertices < 15:
            shape_name = f"{vertices}-sided Polygon"
            
        # -----------------------------------------------------------------
        # DRAW RESULTS AND DISPLAY METRICS
        # -----------------------------------------------------------------
        # Draw the target shape's main contour outline in solid Green
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
        
        # Draw the blue bounding box container around the shape
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # Overlay classification label and metrics right next to the shape
        cv2.putText(frame, shape_name, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Area: {int(area)}px", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Peri: {int(perimeter)}px", (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Print geometric data out in the terminal console for submission verification
        print(f"Detected: {shape_name} | Area: {area:.1f} | Perimeter: {perimeter:.1f} | Box: X:{x} Y:{y} W:{w} H:{h}")
        
    return frame


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if USE_WEBCAM:
    # Open connection to default laptop camera
    cap = cv2.VideoCapture(0)
    print("Starting Live Video Mode... Click on window and press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame)
        cv2.imshow("Live Geometric Shape Detector", processed_frame)
        
        # Keep window running until 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
else:
    # Run pipeline on a static image file instead
    print(f"Reading image file: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print("Error: Could not read image file. Please check your path and filename!")
    else:
        processed_image = process_frame(image)
        cv2.imshow("Image Geometric Shape Detector", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
