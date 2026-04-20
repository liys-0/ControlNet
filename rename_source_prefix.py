import os
import glob

def rename_source_images(dataset_dir):
    source_dir = os.path.join(dataset_dir, "source")
    target_dir = os.path.join(dataset_dir, "target")
    
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        print(f"Error: Make sure both 'source' and 'target' directories exist in {dataset_dir}")
        return

    target_files = glob.glob(os.path.join(target_dir, "*.png"))
    if not target_files:
        print(f"No png files found in {target_dir}")
        return

    renamed_count = 0
    skipped_count = 0
    missing_count = 0

    for target_path in target_files:
        target_filename = os.path.basename(target_path)
        
        if target_filename.startswith("defect_"):
            prefix = "defect_"
            base_name = target_filename[len("defect_"):]
        elif target_filename.startswith("normal_"):
            prefix = "normal_"
            base_name = target_filename[len("normal_"):]
        else:
            skipped_count += 1
            continue

        old_source_path = os.path.join(source_dir, base_name)
        new_source_path = os.path.join(source_dir, target_filename)

        if os.path.exists(old_source_path):
            os.rename(old_source_path, new_source_path)
            print(f"Renamed: {base_name} -> {target_filename}")
            renamed_count += 1
        elif os.path.exists(new_source_path):
            skipped_count += 1
        else:
            print(f"Warning: Source file {base_name} not found!")
            missing_count += 1

    print("-" * 30)
    print(f"Finished processing {len(target_files)} target files.")
    print(f"Successfully renamed: {renamed_count}")
    print(f"Already had prefix or skipped: {skipped_count}")
    print(f"Missing source files: {missing_count}")

if __name__ == "__main__":
    DATASET_DIR = "/home/lys/projects/POC_Dataset/for_ControlNet_all"
    rename_source_images(DATASET_DIR)
