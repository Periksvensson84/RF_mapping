import cv2
import os
import numpy as np

# Global variables to store the selected point
selected_point = None

def select_point(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        cv2.destroyAllWindows()

def convert_video(input_path, crop_x=1274, crop_y=720, target_fps=30):
    global selected_point

    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # Get the directory and file name from the input path
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    # Construct the output path in the same directory
    output_path = os.path.join(input_dir, f"{input_filename}_converted.mp4")

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video Properties: FPS={original_fps}, Width={original_width}, Height={original_height}")

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    # Show the first frame and set up the mouse callback to select the point
    cv2.imshow("Select ROI Center", first_frame)
    cv2.setMouseCallback("Select ROI Center", select_point)
    cv2.waitKey(0)

    if selected_point is None:
        print("Error: No point selected.")
        return

    center_x, center_y = selected_point

    # Calculate frame skipping ratio
    frame_skip_ratio = original_fps / target_fps

    # Set the codec and create VideoWriter object5rf4e n
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_x, crop_y))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Only process every nth frame based on the frame skip ratio
        if frame_count % frame_skip_ratio < 1:
            # Calculate top-left corner of the crop to center it around the selected point
            start_x = max(center_x - crop_x // 2, 0)
            start_y = max(center_y - crop_y // 2, 0)

            # Ensure the crop area does not exceed the frame boundaries
            end_x = min(start_x + crop_x, original_width)
            end_y = min(start_y + crop_y, original_height)

            # Crop the frame
            cropped_frame = frame[start_y:end_y, start_x:end_x]

            # Create a black canvas of the target size
            canvas = np.zeros((crop_y, crop_x, 3), dtype=np.uint8)

            # Calculate the position to place the cropped frame on the canvas
            canvas_start_x = max((crop_x - (end_x - start_x)) // 2, 0)
            canvas_start_y = max((crop_y - (end_y - start_y)) // 2, 0)

            # Place the cropped frame on the canvas
            canvas[canvas_start_y:canvas_start_y + (end_y - start_y), canvas_start_x:canvas_start_x + (end_x - start_x)] = cropped_frame

            # Write the frame to the output video
            out.write(canvas)

        frame_count += 1

    # Release everything when job is finished
    cap.release()
    out.release()
    print(f"Converted video saved to: {output_path}")

# Example usage:
#input_video = r'C:\Python Programming\LIU\Data\Videos\2025-03-27_training-dataset_monocolors\VID20250327135546.mp4'
#convert_video(input_video)

# convert all videos in a directory
input_dir = r'C:\Python Programming\LIU\Data\Videos\2025-03-27_training-dataset_monocolors'
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        input_video = os.path.join(input_dir, filename)
        convert_video(input_video)
        print(f"Converted {filename} to {filename}_converted.mp4")
print("All videos converted.")