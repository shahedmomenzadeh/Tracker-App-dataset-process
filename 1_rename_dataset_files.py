import os
import argparse
import re

def find_corresponding_label(label_dir, image_basename):
    """
    Finds the label file that starts with the image's base name,
    accommodating for random suffixes.
    """
    if not os.path.isdir(label_dir):
        return None
    for label_filename in os.listdir(label_dir):
        if label_filename.startswith(image_basename) and label_filename.lower().endswith('.txt'):
            return os.path.join(label_dir, label_filename)
    return None

def natural_sort_key(s):
    """
    Provides a key for natural sorting (e.g., 'frame_2' before 'frame_10').
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_and_rename_all(root_directory):
    """
    Processes each subfolder in the root directory. It first renames the
    subfolder based on the video name, then renames the image and label
    files within it to a clean, zero-padded numerical format (e.g., 000001.jpg).
    """
    print(f"🚀 Starting Step 1: Renaming and Cleaning in directory: {root_directory}")

    try:
        # Get a list of all items that are directories
        subfolder_names = [f.name for f in os.scandir(root_directory) if f.is_dir()]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{root_directory}' was not found.")
        return

    for folder_name in sorted(subfolder_names):
        # Skip special/output folders
        if folder_name.startswith('.') or folder_name in ['dataset', 'Pupil Labels', 'visualized_videos']:
            continue
            
        print(f"\n--- Processing raw folder: {folder_name} ---")
        original_folder_path = os.path.join(root_directory, folder_name)
        image_dir = os.path.join(original_folder_path, 'images')
        label_dir = os.path.join(original_folder_path, 'labels')

        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            print(f"  ⚠️  Skipping '{folder_name}' (missing 'images' or 'labels' subfolder).")
            continue

        # --- Check if already processed ---
        image_files_check = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files_check:
            print(f"  - ❗️ Warning: No images found. Skipping.")
            continue
        
        first_img_basename = os.path.splitext(image_files_check[0])[0]
        if first_img_basename.isdigit():
            print(f"  ✅ Files appear to be already renamed. Skipping folder.")
            continue

        # --- 1. FOLDER RENAMING ---
        try:
            # Sort files naturally to get the true first frame
            sorted_images = sorted(image_files_check, key=natural_sort_key)
            first_image_basename = os.path.splitext(sorted_images[0])[0]
            
            if '_frame_' not in first_image_basename:
                print(f"  - ❗️ Warning: Image name format incorrect in '{sorted_images[0]}'. Skipping.")
                continue
            
            new_folder_name = first_image_basename.split('_frame_')[0]
            new_folder_path = os.path.join(root_directory, new_folder_name)
            
            # Rename the folder if necessary
            if original_folder_path != new_folder_path:
                os.rename(original_folder_path, new_folder_path)
                print(f"  -> Renamed Folder: '{folder_name}' -> '{new_folder_name}'")
            
            # Update paths to reflect the new folder name
            current_folder_path = new_folder_path
            current_image_dir = os.path.join(current_folder_path, 'images')
            current_label_dir = os.path.join(current_folder_path, 'labels')

        except Exception as e:
            print(f"  - ❌ Error renaming folder '{folder_name}': {e}. Skipping.")
            continue
        
        # --- 2. FILE RENAMING ---
        print("  -> Renaming image and label files...")
        image_files_to_rename = sorted(os.listdir(current_image_dir), key=natural_sort_key)

        for image_filename in image_files_to_rename:
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name, img_ext = os.path.splitext(image_filename)
            old_image_path = os.path.join(current_image_dir, image_filename)
            old_label_path = find_corresponding_label(current_label_dir, base_name)

            if not old_label_path:
                print(f"  - ❗️ Warning: Label for '{image_filename}' not found. Skipping pair.")
                continue

            try:
                # Extract frame number, even with suffixes
                frame_part = base_name.split('_frame_')[1]
                frame_number = int(frame_part.split('_')[0])
                new_base_name = f"{(frame_number + 1):06d}" # Frames start from 1
                
                new_image_path = os.path.join(current_image_dir, new_base_name + img_ext)
                new_label_path = os.path.join(current_label_dir, new_base_name + '.txt')

                os.rename(old_image_path, new_image_path)
                os.rename(old_label_path, new_label_path)

            except (ValueError, IndexError):
                print(f"  - ❗️ Warning: Could not extract frame number from '{base_name}'. Skipping.")
            except OSError as e:
                print(f"  - ❌ Error renaming file '{base_name}': {e}")
                
    print(f"\n✨ Step 1 Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Rename raw data folders and their internal image/label files."
    )
    parser.add_argument(
        "directory", 
        type=str, 
        default=".",
        nargs="?",
        help="The root directory containing the folders to process (defaults to current directory)."
    )
    args = parser.parse_args()
    process_and_rename_all(args.directory)

