import cv2
import os
import numpy as np
from collections import deque
import time

def variance_of_laplacian(image):
    
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Calculate the variance of the Laplacian
    variance = laplacian.var()
    return variance

def analyze_video_clarity(video_path, output_folder, clarity_threshold_percent=85.0):
    # Extract frames from a drone video (handled by VideoCapture)
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

   
    frame_window = deque(maxlen=3)
    frame_number = 0
    saved_count = 0

    print("Starting video analysis...")
    while True:
        grabbed, frame = stream.read()
        if not grabbed:
            break # End of video

        # Measure blurriness of each frame
        sharpness_score = variance_of_laplacian(frame)
        
        # Add the new frame's info to our sliding window
        frame_window.append((frame_number, sharpness_score, frame))

        # Compare each frame with its previous and next frames
        # We can only do this when our window is full (has 3 frames)
        if len(frame_window) == 3:
            # The frame to analyze is the one in the middle of the window
            prev_frame_info = frame_window[0]
            current_frame_info = frame_window[1]
            next_frame_info = frame_window[2]

            # Get the sharpness scores
            prev_score = prev_frame_info[1]
            current_score = current_frame_info[1]
            next_score = next_frame_info[1]
            
            # Calculate the average sharpness of the neighbors
            avg_neighbor_score = (prev_score + next_score) / 2.0
            
            
            if avg_neighbor_score > 0:
                # Compute a clarity percentage score
                clarity_percentage = (current_score / avg_neighbor_score) * 100
            else:
                # If neighbors have no sharpness, any sharpness is an improvement
                clarity_percentage = 100.0 if current_score > 0 else 0.0

            #Save blurry frames (low-score ones)
            if clarity_percentage < clarity_threshold_percent:
                saved_count += 1
                
                
                blurry_frame_number = current_frame_info[0]
                blurry_frame_data = current_frame_info[2]
                
                
                filename = f"frame_{blurry_frame_number:06d}_clarity_{clarity_percentage:.2f}%.jpg"
                filepath = os.path.join(output_folder, filename)
                
                cv2.imwrite(filepath, blurry_frame_data)
                print(f"  -> Saved blurry frame {blurry_frame_number} with clarity: {clarity_percentage:.2f}%")

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"...processed {frame_number} frames...")

    stream.release()
    print("\nAnalysis complete.")
    print(f"Total frames processed: {frame_number}")
    print(f"Total blurry frames saved: {saved_count}")


if __name__ == "__main__":
    VIDEO_FILE = "input_video.mp4"
    OUTPUT_DIR = "drone_blurry_frames"
    CLARITY_THRESHOLD = 85.0 

    
    if not os.path.exists(VIDEO_FILE):
        print(f"'{VIDEO_FILE}' not found. Creating a dummy video for demonstration.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_FILE, fourcc, 30.0, (1280, 720))
        for i in range(150):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            text = f'Frame {i}'
            
            
            if 50 <= i < 55 or 100 <= i < 105:
                sharp_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                frame = cv2.GaussianBlur(sharp_frame, (45, 45), 0)
                text += ' (Intentionally Blurry)'
            else:                 
                 cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            out.write(frame)
        out.release()
        print("Dummy video created successfully.")

    
    start_time = time.time()
    analyze_video_clarity(VIDEO_FILE, OUTPUT_DIR, CLARITY_THRESHOLD)
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")