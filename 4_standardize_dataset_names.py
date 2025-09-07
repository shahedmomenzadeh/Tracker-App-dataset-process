import os
import json
import argparse

def standardize_dataset_names(dataset_root):
    """
    Renames dataset folders, image files, and updates the corresponding
    annotation.json file to a standardized format.

    - Folder: 'TR_0157_4533_S1_P03' -> 'TR_0157_S1_P03'
    - Image:  '000001.jpg' -> 'TR_0157_S1_P03_000001.jpg'
    - JSON:   Updates description and frame filenames.
    """
    print(f"🚀 Starting Step 4: Standardizing names in directory: {dataset_root}")

    if not os.path.isdir(dataset_root):
        print(f"❌ Error: The dataset directory '{dataset_root}' was not found.")
        return

    # Get a sorted list of subdirectories to process
    try:
        subfolder_names = sorted([f.name for f in os.scandir(dataset_root) if f.is_dir()])
    except FileNotFoundError:
        print(f"❌ Error: Could not scan directory '{dataset_root}'.")
        return

    for folder_name in subfolder_names:
        original_folder_path = os.path.join(dataset_root, folder_name)
        print(f"\n--- Processing folder: {folder_name} ---")

        # --- 1. Determine New Folder Name & Skip if Already Processed ---
        parts = folder_name.split('_')
        # Standard format is TR_XXXX_SX_PXX (4 parts)
        # Old format is TR_XXXX_XXXX_SX_PXX (5 parts)
        if len(parts) == 4:
            print("  ✅ Folder name appears to be already standardized. Skipping.")
            continue
        elif len(parts) != 5:
            print(f"  ⚠️  Warning: Folder name '{folder_name}' does not match expected format. Skipping.")
            continue
        
        # Create the new name by removing the third part (index 2)
        new_folder_name = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}"
        new_folder_path = os.path.join(dataset_root, new_folder_name)

        print(f"  -> New standard name will be: {new_folder_name}")

        # --- 2. Rename the Folder ---
        try:
            os.rename(original_folder_path, new_folder_path)
            print(f"  -> Renamed folder to '{new_folder_name}'")
        except OSError as e:
            print(f"  - ❌ Error renaming folder: {e}. Skipping.")
            continue

        # --- 3. Update the annotation.json file ---
        json_path = os.path.join(new_folder_path, "annotation.json")
        if not os.path.exists(json_path):
            print(f"  - ❗️ Warning: 'annotation.json' not found in new folder path. Skipping JSON/image updates.")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Update description
            data['info']['description'] = f"Annotations for {new_folder_name}"

            # Update frame file names
            original_filenames = data['videos'][0]['file_names']
            new_filenames = [f"{new_folder_name}_{os.path.basename(f)}" for f in original_filenames]
            data['videos'][0]['file_names'] = new_filenames

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"  -> Updated 'annotation.json' with {len(new_filenames)} new frame names.")

        except (IOError, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  - ❌ Error updating JSON file: {e}. Cannot proceed with image renaming.")
            continue

        # --- 4. Rename the Image Files ---
        try:
            print("  -> Renaming image files...")
            # We use the original filenames list as the source of truth for renaming
            for original_filename, new_filename in zip(original_filenames, new_filenames):
                old_image_path = os.path.join(new_folder_path, original_filename)
                new_image_path = os.path.join(new_folder_path, new_filename)
                
                if os.path.exists(old_image_path):
                    os.rename(old_image_path, new_image_path)
                else:
                    print(f"    - ❗️ Warning: Source image '{original_filename}' not found. Cannot rename.")
            print("  -> Image renaming complete.")

        except OSError as e:
            print(f"  - ❌ Error renaming image files: {e}.")

    print("\n✨ Step 4 Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Standardize dataset folder names, image names, and JSON annotations."
    )
    parser.add_argument(
        "dataset_directory", 
        type=str, 
        default="./dataset",
        nargs="?",
        help="The root directory of the processed dataset (defaults to './dataset')."
    )
    args = parser.parse_args()
    standardize_dataset_names(args.dataset_directory)

