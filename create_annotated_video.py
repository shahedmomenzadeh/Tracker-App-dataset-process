import cv2
import numpy as np
import json
import os
import glob
import argparse

# --- Configuration ---
DATASET_ROOT = "dataset/"
VIDEO_OUTPUT_DIR = "visualized_videos/"

# Define a color dictionary for consistent class colors
COLOR_DICT = {
    "Cannula": (255, 0, 0), "Cap-Cystotome": (0, 255, 0), "Cap-Forceps": (0, 0, 255),
    "Cornea": (255, 255, 0), "Forceps": (255, 0, 255), "IA-Handpiece": (0, 255, 255),
    "Lens-Injector": (125, 125, 0), "Phaco-Handpiece": (0, 125, 125), "Primary-Knife": (125, 0, 125),
    "Pupil": (50, 200, 200), "Second-Instrument": (200, 200, 50), "Secondary-Knife": (200, 50, 200),
    "Default": (128, 128, 128)
}

# --- Drawing Functions ---


# Store trajectories: {class_name: [(x1, y1), (x2, y2), ...]}
TRAJECTORIES = {}


def draw_annotations_on_frame(frame, annotations_for_frame):
    overlay = frame.copy()
    
    for ann in annotations_for_frame:
        class_name = ann['class_name']
        color = COLOR_DICT.get(class_name, COLOR_DICT["Default"])
        
        # Draw segmentation mask
        if ann['segmentation']:
            try:
                poly = np.array(ann['segmentation'][0], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [poly], color)
            except (ValueError, IndexError):
                pass

        # Draw bounding box
        if ann['bbox']:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Draw keypoints
        if ann['keypoints']:
            # Center (Pupil, Cornea)
            if ann['keypoints'][2] == 2: 
                center_x, center_y = int(ann['keypoints'][0]), int(ann['keypoints'][1])
                cv2.circle(frame, (center_x, center_y), 6, (255, 0, 0), -1, cv2.LINE_AA)

            # Tip (for instruments)
            if ann['keypoints'][5] == 2: 
                tip_x, tip_y = int(ann['keypoints'][3]), int(ann['keypoints'][4])
                cv2.circle(frame, (tip_x, tip_y), 6, color, -1, cv2.LINE_AA)  # Use class color

                # Save trajectory except for Cornea and Pupil
                if class_name not in ["Cornea", "Pupil"]:
                    if class_name not in TRAJECTORIES:
                        TRAJECTORIES[class_name] = []
                    TRAJECTORIES[class_name].append((tip_x, tip_y))

    # Draw trajectories for all classes
    for cname, points in TRAJECTORIES.items():
        if len(points) > 1:
            clr = COLOR_DICT.get(cname, COLOR_DICT["Default"])
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], clr, 2)

    cv2.addWeighted(overlay, 0.5, frame, 0.6, 0, frame)
    return frame


# --- Main Processing Function ---

def create_video(video_folder_path, json_filename):
    """
    Creates an annotated video from a folder of frames and a JSON file.
    """
    video_name = os.path.basename(video_folder_path)
    input_json_path = os.path.join(video_folder_path, json_filename)
    
    if not os.path.exists(input_json_path):
        print(f"  [Error] Annotation file not found: {input_json_path}. Skipping video creation.")
        return

    print(f"  -> Loading annotations from: {json_filename}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Prepare data structure for easy frame-by-frame access
    num_frames = len(data['videos'][0]['file_names'])
    all_frames_data = [[] for _ in range(num_frames)]
    
    for ann in data['annotations']:
        class_name = category_map.get(ann['category_id'], "Unknown")
        # All categories have ["center", "tip"], so 2 keypoints. Stride is 2 * 3 = 6.
        kp_stride = 6 

        for i in range(num_frames):
            # Check for the presence of a bounding box to determine if the object exists on this frame.
            # This makes the script compatible with datasets that don't have segmentation masks.
            if ann['bboxes'] and i < len(ann['bboxes']) and ann['bboxes'][i]:
                frame_ann = {
                    'class_name': class_name,
                    # Segmentation can be None, so we handle that gracefully
                    'segmentation': ann['segmentations'][i] if ann['segmentations'] and i < len(ann['segmentations']) else None,
                    'bbox': ann['bboxes'][i],
                    'keypoints': ann['keypoints'][i * kp_stride : (i + 1) * kp_stride]
                }
                all_frames_data[i].append(frame_ann)

    # --- Video Creation ---
    output_video_name = f"{video_name}_visualized.mp4"
    output_video_path = os.path.join(VIDEO_OUTPUT_DIR, output_video_name)
    
    width = data['videos'][0]['width']
    height = data['videos'][0]['height']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    print(f"  -> Creating video: {output_video_name}")
    for i, frame_filename in enumerate(data['videos'][0]['file_names']):
        frame_path = os.path.join(video_folder_path, frame_filename)
        if not os.path.exists(frame_path):
            print(f"    [Warning] Frame not found: {frame_path}")
            continue
            
        frame = cv2.imread(frame_path)
        annotations_for_frame = all_frames_data[i]
        
        annotated_frame = draw_annotations_on_frame(frame, annotations_for_frame)
        
        out_writer.write(annotated_frame)
        print(f"    Processing frame {i+1}/{num_frames}", end='\r')
        
    out_writer.release()
    print(f"\n  -> Successfully created video: {output_video_path}")


# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(
        description="Create annotated videos from dataset frames and a specified JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=DATASET_ROOT,
        help="Path to the root directory of the dataset."
    )
    parser.add_argument(
        "--videos", nargs='+', default=None,
        help="Optional: Specify one or more video names (e.g., 'SK_0002_1006_S1_P03').\nIf not provided, all videos in the dataset directory will be processed."
    )
    parser.add_argument(
        "--json_file", type=str, default="annotation.json",
        help="The exact name of the annotation file to use (e.g., 'annotation.json')."
    )
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(args.dataset_dir):
        print(f"[Error] Input dataset directory not found at '{args.dataset_dir}'")
        return

    # --- Determine which videos to process ---
    video_names = []
    if args.videos:
        # If a list of videos is provided via the command line, use that list.
        video_names = args.videos
        print(f"Processing {len(video_names)} specified video(s): {', '.join(video_names)}")
    else:
        # If no specific videos are provided, find all subdirectories in the dataset folder.
        print(f"No specific videos provided. Searching for all video datasets in '{args.dataset_dir}'...")
        video_names = [os.path.basename(d) for d in glob.glob(os.path.join(args.dataset_dir, '*')) if os.path.isdir(d)]
        if video_names:
            print(f"Found {len(video_names)} video dataset(s) to process.")

    if not video_names:
        print(f"No video subdirectories found to process in '{args.dataset_dir}'.")
        return

    # --- Processing Loop ---
    for video_name in video_names:
        video_folder_path = os.path.join(args.dataset_dir, video_name)
        
        # Check if the specific video folder exists before processing
        if not os.path.isdir(video_folder_path):
            print(f"\n[Warning] Could not find directory for video '{video_name}'. Skipping.")
            continue
            
        print(f"\n--- Processing video: {video_name} ---")
        create_video(video_folder_path, args.json_file)
        print(f"--- Finished video creation for {video_name} ---")

if __name__ == "__main__":
    main()

