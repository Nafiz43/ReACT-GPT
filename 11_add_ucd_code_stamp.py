import os

# Define the header content
HEADER_TEXT = {
    '.py': '''"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
''',

    '.sh': '''# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
''',

    '.html': '''<!--
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
-->
''',
    '.js': '''// Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
// Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
''',
}

SUPPORTED_EXTENSIONS = set(HEADER_TEXT.keys())

def insert_header(file_path, ext):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    header = HEADER_TEXT[ext]
    if header.strip() in content:
        print(f"✔️ Skipped (already has header): {file_path}")
        return

    # Handle shebang for scripts
    lines = content.splitlines(keepends=True)
    if lines and lines[0].startswith('#!'):
        shebang = lines[0]
        rest = ''.join(lines[1:])
        new_content = shebang + header + rest
    else:
        new_content = header + content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ Header added to: {file_path}")

def process_directory(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                insert_header(file_path, ext)

if __name__ == "__main__":
    dir_path = ('/home/nafiz/Documents/ReACT-GPT').strip()
    if not os.path.isdir(dir_path):
        print("❌ Invalid directory.")
    else:
        process_directory(dir_path)
