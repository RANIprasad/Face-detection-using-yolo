from ultralytics import YOLO
import cv2

# Load the trained face detection model
model = YOLO(r"C:\Users\sumit\Desktop\yolo\proj\data\runs\detect\train4\weights\best.pt")

# Path to the video file
video_path = r"C:\Users\sumit\Desktop\yolo\proj\data\test_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot access the video.")
    exit()

# Get the video's width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Resize the frames to match the model's expected input size (640x640)
resized_width = 640
resized_height = 640

# Create a VideoWriter object to save the output
output_path = r"C:\Users\sumit\Desktop\yolo\proj\data\output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (for .mp4 format)
out = cv2.VideoWriter(output_path, fourcc, fps, (resized_width, resized_height))

if not out.isOpened():
    print("Error: Failed to open the video writer.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    
    if not ret:
        print("Failed to grab frame or end of video.")
        break

    resized_frame = cv2.resize(frame, (640, 640))  # Resize frame to match model input size

    # Perform inference on the frame
    results = model.predict(resized_frame, conf=0.5)  # Adjust confidence threshold as needed

    # Visualize predictions on the frame
    annotated_frame = results[0].plot()  # Overlay results on the frame

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Video saved to {output_path}")
