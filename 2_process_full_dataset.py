import os
import math
import glob
import yaml

# --- Utility Functions ---

def find_pupil_label_file(pupil_dir, base_name):
    """Finds the pupil label file for frame 0."""
    search_prefix = f"{base_name}_frame_0"
    if not os.path.isdir(pupil_dir):
        return None
    for filename in os.listdir(pupil_dir):
        if filename.startswith(search_prefix) and filename.endswith('.txt'):
            return os.path.join(pupil_dir, filename)
    return None

def parse_yolo_segmentation_line(line):
    """Parses a YOLO segmentation line into class index and points."""
    parts = line.strip().split()
    if len(parts) < 3: return None, None
    try:
        return int(parts[0]), [float(p) for p in parts[1:]]
    except (ValueError, IndexError):
        return None, None

def get_bounding_box_from_points(points):
    """Calculates the width and height of the bounding box from segmentation points."""
    if not points or len(points) % 2 != 0: return 0, 0
    x_coords, y_coords = points[0::2], points[1::2]
    return max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

def get_centroid_from_points(points):
    """Calculates the geometric center (centroid) of a polygon."""
    if not points or len(points) % 2 != 0: return 0, 0
    x_coords, y_coords = points[0::2], points[1::2]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

# --- Main Processing Logic ---

def process_pupil_masks_for_folder(folder_path, pupil_labels_dir):
    """
    Adds pupil mask to the first frame, updates the YAML file, and propagates
    the mask to all other frames.
    """
    folder_name = os.path.basename(folder_path)
    labels_dir = os.path.join(folder_path, 'labels')
    yaml_path = os.path.join(folder_path, 'data.yaml')
    print(f"\n--- Processing folder for pupil masks: {folder_name} ---")
    
    # --- Part 1: Add pupil mask to Frame 1 (000001.txt) ---
    frame_1_path = os.path.join(labels_dir, '000001.txt')
    if not os.path.exists(frame_1_path) or not os.path.exists(yaml_path):
        print(f"  ⚠️  Warning: Cleaned label/YAML file not found. Skipping.")
        return

    pupil_source_file = find_pupil_label_file(pupil_labels_dir, folder_name)
    if not pupil_source_file:
        print(f"  ⚠️  Warning: Pupil source file not found for '{folder_name}'. Skipping.")
        return
        
    # Add pupil mask if it's not already there
    mask_was_added = False
    with open(frame_1_path, 'r+') as f:
        lines = f.readlines()
        if not any(line.strip().startswith('2 ') for line in lines):
            print("  -> Adding pupil mask to Frame 1...")
            mask_was_added = True
            with open(pupil_source_file, 'r') as f_pupil:
                pupil_lines_to_add = []
                for pupil_line in f_pupil:
                    if pupil_line.strip().startswith('0 '): # Pupil is class 0 in source files
                        # Rewrite line with class 2
                        new_line = f"2 {' '.join(pupil_line.strip().split()[1:])}"
                        pupil_lines_to_add.append(new_line)
                if pupil_lines_to_add:
                    f.seek(0, 2) # Go to the end of the file
                    f.write('\n' + '\n'.join(pupil_lines_to_add))
    
    # --- Part 2: Update YAML file if a mask was just added ---
    if mask_was_added:
        try:
            with open(yaml_path, 'r') as f_yaml:
                data = yaml.safe_load(f_yaml)
            
            if 'names' not in data: data['names'] = {}

            if 2 not in data['names']:
                data['names'][2] = 'Pupil'
                with open(yaml_path, 'w') as f_yaml:
                    yaml.dump(data, f_yaml, sort_keys=False, default_flow_style=False)
                print("  -> ✅ Updated YAML file to include 'Pupil' class.")
        except Exception as e:
            print(f"  - ❌ Error updating YAML file: {e}")

    # --- Part 3: Propagate pupil masks to other frames ---
    print("  -> Propagating pupil masks to subsequent frames...")
    
    # Get the scaling ratio from Frame 1
    cornea_f1_pts, pupil_f1_pts = None, None
    with open(frame_1_path, 'r') as f:
        for line in f:
            class_id, points = parse_yolo_segmentation_line(line)
            if class_id == 0: cornea_f1_pts = points
            elif class_id == 2: pupil_f1_pts = points

    if not cornea_f1_pts or not pupil_f1_pts:
        print("  ⚠️  Warning: Cannot propagate. Cornea or Pupil mask missing in Frame 1.")
        return

    cornea_w, cornea_h = get_bounding_box_from_points(cornea_f1_pts)
    pupil_w, pupil_h = get_bounding_box_from_points(pupil_f1_pts)
    cornea_area = cornea_w * cornea_h
    pupil_area = pupil_w * pupil_h

    if cornea_area == 0:
        print("  ⚠️  Warning: Cornea in Frame 1 has zero area. Cannot calculate ratio.")
        return
    
    scale_factor = math.sqrt(pupil_area / cornea_area)
    print(f"  ℹ️ Calculated scale factor from Frame 1: {scale_factor:.4f}")

    # Process all other label files
    for label_path in glob.glob(os.path.join(labels_dir, "*.txt")):
        if os.path.basename(label_path) == '000001.txt':
            continue

        cornea_current_pts = None
        has_pupil = False
        with open(label_path, 'r') as f:
            for line in f:
                class_id, points = parse_yolo_segmentation_line(line)
                if class_id == 2:
                    has_pupil = True
                    break
                elif class_id == 0:
                    cornea_current_pts = points
        
        if has_pupil or not cornea_current_pts:
            continue

        # Create and append the new pupil mask
        centroid_x, centroid_y = get_centroid_from_points(cornea_current_pts)
        new_pupil_points = [p for (px, py) in zip(cornea_current_pts[0::2], cornea_current_pts[1::2]) for p in (centroid_x + (px - centroid_x) * scale_factor, centroid_y + (py - centroid_y) * scale_factor)]

        new_pupil_line = "2 " + " ".join([f"{p:.6f}" for p in new_pupil_points])
        with open(label_path, 'a') as f:
            f.write('\n' + new_pupil_line)


if __name__ == "__main__":
    main_directory = '.'
    pupil_dir = 'Pupil Labels'
    
    print("🚀 Starting Step 2: Adding and Propagating Pupil Masks...")
    
    folders_to_process = [f.path for f in os.scandir(main_directory) if f.is_dir() and not f.name.startswith('.') and f.name not in ['dataset', pupil_dir, 'visualized_videos']]

    if not os.path.isdir(pupil_dir):
        print(f"❌ Error: '{pupil_dir}' directory not found. Cannot proceed.")
    else:
        for folder_path in sorted(folders_to_process):
            process_pupil_masks_for_folder(folder_path, pupil_dir)
        print("\n✨ Step 2 Complete!")

