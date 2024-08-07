from pathlib import Path
import os

video_name = 'gam3974357c68.mp4'
video_path = Path.home() / 'rps' / video_name
    

# Define the base cache directory
base_cache_dir = Path.home() / 'Documents' / 'basket_detection_cache'

# Define cache directories
base_cache_dir = Path.home() / 'Documents' / 'basket_detection_cache'
png_cache_dir = base_cache_dir / 'png_files'
index_path = base_cache_dir / f"{os.path.splitext(video_name)[0]}_index_cv2.json"

# Create cache directories if they don't exist
base_cache_dir.mkdir(parents=True, exist_ok=True)
png_cache_dir.mkdir(parents=True, exist_ok=True)