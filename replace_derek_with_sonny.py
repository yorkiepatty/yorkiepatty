import os
import re

def replace_in_file(filepath, old, new):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    # Replace whole words only, matches "Derek" and "derek" (case-insensitive)
    new_content = re.sub(r'\bDerek\b', 'Sonny', content)
    new_content = re.sub(r'\bderek\b', 'sonny', new_content)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(new_content)

def replace_in_project(root_dir, old, new):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):  # Scan Python files only
                filepath = os.path.join(dirpath, filename)
                replace_in_file(filepath, old, new)
                print(f"Updated: {filepath}")

# Usage
replace_in_project(".", "Derek", "Sonny")
