import os
from pathlib import Path

# Utility to save input files
def save_file(file_content, file_name: str, directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    return file_path

# Utility to retrieve files
def get_file_path(file_name: str, directory: str):
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"File {file_name} not found in {directory}")
