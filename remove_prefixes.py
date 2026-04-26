import os
import argparse

def remove_prefixes(directory):
    prefixes = ["normal_", "defect_"]
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    renamed_count = 0
    
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        
        if not os.path.isfile(old_path):
            continue
            
        new_name = filename
        
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
                
        if new_name != filename:
            new_path = os.path.join(directory, new_name)
            
            if os.path.exists(new_path):
                print(f"Warning: Cannot rename '{filename}' to '{new_name}' because '{new_name}' already exists. Skipping.")
                continue
                
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            renamed_count += 1
            
    print(f"\nDone! Total files renamed: {renamed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'normal_' and 'defect_' prefixes from filenames in a folder.")
    parser.add_argument("directory", help="The path to the target folder")
    
    args = parser.parse_args()
    remove_prefixes(args.directory)
