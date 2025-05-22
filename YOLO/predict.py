import cv2
from ultralytics import YOLO
from pathlib import Path

# Load a model
# model = YOLO("runs/detect/train10/weights/best.pt")  # pretrained YOLO model
model = YOLO("yolov12m.pt")  # pretrained YOLO model

# Define path to video file
source = "kinect-training-set.mkv"
output_filename = "output2.mp4"

# --- VideoWriter Setup ---
# Open the source video to get properties
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error opening video file: {source}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release() # Release the capture object, we'll get frames from YOLO

# Define the codec and create VideoWriter object
# Use 'mp4v' or 'avc1' for .mp4, check compatibility if needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
# --- End VideoWriter Setup ---


# Run batched inference on the video
# Set stream=True for memory efficiency with long videos
results = model(source, stream=True)

# Process results generator
print(f"Processing video: {source}")
print(f"Saving output to: {output_filename}")
for result in results:
    # Visualize the results on the frame
    annotated_frame = result.plot()

    # Write the frame to the output video file
    out.write(annotated_frame)

    # Optional: display the frame to screen
    # cv2.imshow("YOLO Inference", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit early
    #    break

# Release the video writer object
out.release()
# Optional: Close display window if used
# cv2.destroyAllWindows()

print("Video processing complete.")
