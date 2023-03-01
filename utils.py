from pathlib import Path
import json

def get_next_filepath(directory, base_filename):
    index = 1
    while True:
        filename = f"{base_filename}_{index}.json"
        filepath = directory / filename
        if not filepath.exists():
            return filepath
        index += 1
