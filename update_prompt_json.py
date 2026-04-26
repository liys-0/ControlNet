import json
import argparse
import os

def update_prompt_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return

    updated_lines = []
    modified_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                modified = False
                
                if 'source' in data:
                    new_source = data['source'].replace('source/normal_', 'source/').replace('source/defect_', 'source/')
                    if new_source != data['source']:
                        data['source'] = new_source
                        modified = True
                        
                if 'target' in data:
                    new_target = data['target'].replace('target/normal_', 'target/').replace('target/defect_', 'target/')
                    if new_target != data['target']:
                        data['target'] = new_target
                        modified = True
                
                if modified:
                    modified_count += 1
                    
                updated_lines.append(json.dumps(data))
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                updated_lines.append(line.strip())

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')
            
    print(f"Successfully updated {modified_count} entries in {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'normal_' and 'defect_' prefixes from source and target paths in prompt.json.")
    parser.add_argument("file_path", help="Path to the prompt.json file")
    
    args = parser.parse_args()
    update_prompt_json(args.file_path)