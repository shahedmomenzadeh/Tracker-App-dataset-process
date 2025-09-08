import cv2
import numpy as np
import json
import os
import glob
import shutil
import time
import yaml

# --- Prerequisite Libraries ---
# pip install opencv-python numpy pyyaml

# --- Custom JSON Encoder ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Configuration ---
VIDEO_BASE_DIR = "./"
DATASET_DIR = "dataset/"
FINAL_CLASSES = [
    "Cannula", "Cap-Cystotome", "Cap-Forceps", "Cornea", "Forceps",
    "IA-Handpiece", "Lens-Injector", "Phaco-Handpiece", "Primary-Knife",
    "Pupil", "Second-Instrument", "Secondary-Knife"
]

# --- Dataset Generation Class ---
class DatasetGenerator:
    def __init__(self, video_name, frame_size, all_classes):
        self.video_name = video_name
        self.width, self.height = frame_size
        self.categories = [{"id": i + 1, "name": name, "keypoints": ["center", "tip"], "skeleton": []}
                           for i, name in enumerate(all_classes)]
        self.class_to_id = {name: i + 1 for i, name in enumerate(all_classes)}
        self.dataset = {
            "info": {"description": f"Annotations for {video_name}", "version": "1.0", "year": time.strftime("%Y")},
            "licenses": [{"id": 1, "name": "ARAS", "url": ""}],
            "categories": self.categories, "videos": [], "annotations": []
        }
    def add_video_entry(self, frame_files): self.dataset["videos"].append({"id": 1, "width": self.width, "height": self.height, "file_names": frame_files})
    def add_annotation(self, instance_id, category_name, segmentations, bboxes, areas, keypoints):
        if category_name not in self.class_to_id: return
        self.dataset["annotations"].append({"id": instance_id, "video_id": 1, "category_id": self.class_to_id[category_name], "iscrowd": 0, "segmentations": segmentations, "bboxes": bboxes, "areas": areas, "keypoints": keypoints})
    def save_json(self, output_path):
        with open(output_path, 'w') as f: json.dump(self.dataset, f, indent=4, cls=NumpyEncoder)

# --- Utility Functions ---
def compute_tool_tip(mask, pupil_center):
    if pupil_center is None or not np.any(mask): return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    all_contour_points = np.vstack(contours)
    distances = np.linalg.norm(all_contour_points.squeeze(axis=1) - pupil_center, axis=1)
    return tuple(all_contour_points[np.argmin(distances)].squeeze().astype(int))

# --- Integrated Standardization Function ---
def standardize_coco_names(output_folder_path):
    """Renames the folder, images, and JSON content to the final standard."""
    print(f"  -> Standardizing names for {os.path.basename(output_folder_path)}...")
    folder_name = os.path.basename(output_folder_path)
    dataset_root = os.path.dirname(output_folder_path)
    
    parts = folder_name.split('_')
    if len(parts) != 5:
        print(f"  -> ✅ Already standardized. Skipping.")
        return

    # Create the new name by removing the third part
    new_folder_name = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}"
    new_folder_path = os.path.join(dataset_root, new_folder_name)
    
    # Rename folder
    os.rename(output_folder_path, new_folder_path)
    
    json_path = os.path.join(new_folder_path, "annotation.json")
    if not os.path.exists(json_path): return

    with open(json_path, 'r') as f: data = json.load(f)
    
    # Update JSON content
    data['info']['description'] = f"Annotations for {new_folder_name}"
    original_filenames = data['videos'][0]['file_names']
    # MODIFIED: New filename format without '_frame_'
    new_filenames = [f"{new_folder_name}_{os.path.basename(f)}" for f in original_filenames]
    data['videos'][0]['file_names'] = new_filenames

    with open(json_path, 'w') as f: json.dump(data, f, indent=4)
    
    # Rename image files
    for original_fn, new_fn in zip(original_filenames, new_filenames):
        old_image_path = os.path.join(new_folder_path, original_fn)
        new_image_path = os.path.join(new_folder_path, new_fn)
        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)
    print(f"  -> ✅ Renamed folder, images, and JSON to '{new_folder_name}' standard.")

# --- Main Processing Function ---
def process_video_folder(video_folder_path):
    video_name = os.path.basename(video_folder_path)
    print(f"\n--- Converting to COCO: {video_name} ---")

    images_dir = os.path.join(video_folder_path, 'images')
    labels_dir = os.path.join(video_folder_path, 'labels')
    yaml_path = os.path.join(video_folder_path, 'data.yaml')
    output_dir = os.path.join(DATASET_DIR, video_name)

    if not all([os.path.isdir(images_dir), os.path.isdir(labels_dir), os.path.exists(yaml_path)]):
        print(f"  ⚠️  Skipping: Missing 'images', 'labels', or 'data.yaml'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(yaml_path, 'r') as f: yaml_data = yaml.safe_load(f)
    class_map = {int(k): v for k, v in yaml_data['names'].items()}
    if 1 in class_map: class_map[1] = "Cap-Cystotome"
    if 2 in class_map: class_map[2] = "Pupil"

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    num_frames = len(label_files)
    if num_frames == 0: return

    first_image_path = glob.glob(os.path.join(images_dir, "000001.*"))[0]
    h, w, _ = cv2.imread(first_image_path).shape
    frame_size = (w, h)

    all_frame_data = {name: [None] * num_frames for name in class_map.values()}
    for i, label_file in enumerate(label_files):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                class_id, pts = int(parts[0]), [float(p) for p in parts[1:]]
                class_name = class_map.get(class_id)
                if not class_name: continue
                poly_abs = (np.array(pts).reshape(-1, 2) * [w, h]).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(mask, [poly_abs], 1); mask = mask.astype(bool)
                all_frame_data[class_name][i] = {'seg': mask, 'bbox': cv2.boundingRect(poly_abs), 'area': np.sum(mask)}

    pupil_centers = [None] * num_frames
    if 'Pupil' in all_frame_data:
        for i in range(num_frames):
            if all_frame_data['Pupil'][i]:
                M = cv2.moments(all_frame_data['Pupil'][i]['seg'].astype(np.uint8))
                if M["m00"] > 0:
                    pupil_centers[i] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    all_frame_data['Pupil'][i]['center'] = pupil_centers[i]
    
    for class_name, frame_list in all_frame_data.items():
        for i, frame_data in enumerate(frame_list):
            if not frame_data: continue
            if class_name not in ['Pupil', 'Cornea']:
                frame_data['tip'] = compute_tool_tip(frame_data['seg'], pupil_centers[i])
            elif class_name == 'Cornea':
                 M = cv2.moments(frame_data['seg'].astype(np.uint8)); frame_data['center'] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] > 0 else None

    dataset_generator = DatasetGenerator(video_name, frame_size, FINAL_CLASSES)
    dataset_generator.add_video_entry([f"{i+1:06d}.jpg" for i in range(num_frames)])
    static_classes = ["Cornea", "Pupil"]
    
    for class_name, track_data in all_frame_data.items():
        if class_name not in FINAL_CLASSES: continue
        segmentations, bboxes, areas, keypoints_flat = [], [], [], []
        for frame_data in track_data:
            if frame_data:
                contours, _ = cv2.findContours(frame_data['seg'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                seg_poly = [p for c in contours for p in c.flatten()] if contours else []
                segmentations.append([seg_poly] if seg_poly else None)
                bboxes.append(frame_data['bbox']); areas.append(frame_data['area'])
                tip, center = frame_data.get('tip'), frame_data.get('center')
                if class_name not in static_classes: keypoints_flat.extend([0, 0, 0, tip[0], tip[1], 2] if tip else [0]*6)
                else: keypoints_flat.extend([center[0], center[1], 2, 0, 0, 0] if center else [0]*6)
            else:
                segmentations.append(None); bboxes.append(None); areas.append(None); keypoints_flat.extend([0]*6)
        dataset_generator.add_annotation(abs(hash(class_name)), class_name, segmentations, bboxes, areas, keypoints_flat)
    
    dataset_generator.save_json(os.path.join(output_dir, "annotation.json"))
    
    print(f"  -> Copying {num_frames} frames...")
    for i, label_file in enumerate(label_files):
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        src_img_path = glob.glob(os.path.join(images_dir, f"{base_name}.*"))[0]
        shutil.copy(src_img_path, os.path.join(output_dir, f"{i+1:06d}.jpg"))

    print(f"  -> Successfully created intermediate COCO dataset for {video_name}.")

    # --- INTEGRATED FINAL STEP ---
    standardize_coco_names(output_dir)

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"🚀 Starting Step 3: Converting YOLO datasets to COCO and Standardizing Names...")
    
    video_folders = [f.path for f in os.scandir(VIDEO_BASE_DIR) if f.is_dir() and not f.name.startswith('.') and f.name not in [DATASET_DIR, 'Pupil Labels', 'visualized_videos']]
    
    for folder_path in sorted(video_folders):
        video_name = os.path.basename(folder_path)
        # Check if the FINAL standardized folder exists to skip
        parts = video_name.split('_')
        if len(parts) == 5:
            final_name = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}"
            if os.path.isdir(os.path.join(DATASET_DIR, final_name)):
                 print(f"\n--- Skipping {video_name} (final dataset already exists) ---")
                 continue
        
        process_video_folder(folder_path)

    print("\n✨ Step 3 Complete!")

if __name__ == "__main__":
    main()

