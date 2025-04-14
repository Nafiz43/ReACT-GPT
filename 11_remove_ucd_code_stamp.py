"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import re

# Define the exact header blocks (must match what was inserted)
HEADER_PATTERNS = {
    '.py': re.compile(
        r'(?s)^"""\s*Developed at DECAL Lab in CS Department @ UC Davis.*?Used with permission\.\s*"""\n*'
    ),

    '.sh': re.compile(
        r'(?m)^# Developed at DECAL Lab in CS Department @ UC Davis.*?Used with permission\.\n*',
    ),

    '.html': re.compile(
        r'(?s)^<!--\s*Developed at DECAL Lab in CS Department @ UC Davis.*?Used with permission\.\s*-->\n*'
    ),
}

SUPPORTED_EXTENSIONS = set(HEADER_PATTERNS.keys())

def remove_header(file_path, ext):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = HEADER_PATTERNS[ext]
    new_content, count = pattern.subn('', content)

    if count == 0:
        print(f"✔️ No header found in: {file_path}")
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ Header removed from: {file_path}")

def process_directory(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                remove_header(file_path, ext)

if __name__ == "__main__":
    dir_path =('/home/nafiz/Documents/ReACT-GPT').strip()
    if not os.path.isdir(dir_path):
        print("❌ Invalid directory.")
    else:
        process_directory(dir_path)
