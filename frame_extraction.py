from os import name
import cv2
from setup import setup_logging, roboflow_setup
import logging
from pathlib import Path
import time
import numpy as np
from typing import Dict, Optional, List, Tuple
import json
import aiofiles
from av import open as av_open



#set up the log file
log_dir = Path.home() / 'Documents' / 'basket_detection_cache'
log_file = log_dir / 'video_processing.log'

setup_logging(log_file)
logger = logging.getLogger(__name__)

#indexing
def index_mp4_with_segments_cv2(file_path: str, output_dir: str = None, segment_duration: int = 300) -> Dict: #good
    """
    Index an MP4 file to extract keyframe information and create segments using OpenCV.
    """
    logger.info(f"Starting to index {file_path} with {segment_duration}-second segments using OpenCV")
    start_time = time.time()
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")

    try:
        # Open the video file
        video = cv2.VideoCapture(str(file_path))
        if not video.isOpened():
            raise IOError("Error opening video file")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video properties: {total_frames} frames, {fps} FPS, duration: {duration:.2f} seconds")
        print(f"Video properties: {total_frames} frames, {fps} FPS, duration: {duration:.2f} seconds")

        keyframes = []
        segments = []
        current_segment = {'start_time': 0, 'end_time': segment_duration, 'keyframes': []}

        print("Starting frame analysis...")
        prev_frame = None
        frame_diffs = []
        for frame_number in range(total_frames):
            ret, frame = video.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_number}")
                break

            timestamp = frame_number / fps

            # Convert frame to grayscale and blur it to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            # Compute frame difference
            frame_diff = cv2.absdiff(prev_frame, gray)
            frame_diff = frame_diff.astype(np.float32).mean()
            frame_diffs.append(frame_diff)

            # Dynamic thresholding: Use mean + 2 * std_dev of recent frame differences
            recent_diffs = frame_diffs[-min(len(frame_diffs), 100):]  # Consider last 100 frames
            threshold = np.mean(recent_diffs) + 2 * np.std(recent_diffs)

            is_keyframe = frame_diff > threshold

            if is_keyframe:
                keyframe_info = {
                    'timestamp': timestamp,
                    'frame_number': frame_number
                }
                keyframes.append(keyframe_info)
                
                if timestamp < current_segment['end_time']:
                    current_segment['keyframes'].append(keyframe_info)
                else:
                    segments.append(current_segment)
                    current_segment = {
                        'start_time': current_segment['end_time'],
                        'end_time': min(current_segment['end_time'] + segment_duration, duration),
                        'keyframes': [keyframe_info]
                    }

            prev_frame = gray

            if frame_number % 100 == 0:
                progress = frame_number / total_frames
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                print(f"\rProgress: {progress:.2%} - Estimated time remaining: {remaining_time:.2f} seconds", end="")
                logger.debug(f"Processed frame {frame_number}, keyframes so far: {len(keyframes)}")

        # Add the last segment if it's not empty
        if current_segment['keyframes']:
            segments.append(current_segment)

        video.release()

        print("\nFrame analysis completed.")
        
        index = {
            'file_path': str(file_path),
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'segment_duration': segment_duration,
            'segments': segments
        }
        
        logger.info(f"Indexed {len(keyframes)} keyframes across {len(segments)} segments")
        print(f"Indexed {len(keyframes)} keyframes across {len(segments)} segments")

        # Save index to JSON file if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            index_filename = f"{file_path.stem}_index_cv2.json"
            output_path = output_dir / index_filename
            
            with output_path.open('w') as f:
                json.dump(index, f, indent=2)
            logger.info(f"Saved index to {output_path}")
            print(f"Saved index to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"Indexing completed in {total_time:.2f} seconds")
        print(f"Indexing completed in {total_time:.2f} seconds")

        return index

    except Exception as e:
        logger.error(f"An error occurred during indexing: {e}")
        raise

class AsyncVideoIndexer: #good
    _instances = {}

    def __new__(cls, video_path: str, index_path: str):
        if video_path not in cls._instances:
            cls._instances[video_path] = super(AsyncVideoIndexer, cls).__new__(cls)
        return cls._instances[video_path]

    def __init__(self, video_path: str, index_path: str):
        if not hasattr(self, 'initialized'):
            self.video_path = video_path
            self.index_path = index_path
            self.index: Optional[Dict] = None
            self.initialized = True

    async def load_index(self):
        if self.index is None:
            try:
                async with aiofiles.open(self.index_path, 'r') as f:
                    content = await f.read()
                    self.index = json.loads(content)
                logger.info(f"Loaded index for {self.video_path}")
            except FileNotFoundError:
                logger.error(f"Index file not found: {self.index_path}")
                raise
        return self.index

    def find_nearest_keyframe(self, target_frame: int) -> int:
        if not self.index:
            raise ValueError("Index not loaded")
        
        keyframes = [kf for segment in self.index['segments'] for kf in segment['keyframes']]
        keyframes.sort(key=lambda x: x['frame_number'])
        
        left, right = 0, len(keyframes) - 1
        while left <= right:
            mid = (left + right) // 2
            if keyframes[mid]['frame_number'] == target_frame:
                return keyframes[mid]['frame_number']
            elif keyframes[mid]['frame_number'] < target_frame:
                left = mid + 1
            else:
                right = mid - 1
        
        if right < 0:
            return keyframes[0]['frame_number']
        if left >= len(keyframes):
            return keyframes[-1]['frame_number']
        
        before = keyframes[right]['frame_number']
        after = keyframes[left]['frame_number']
        return before if target_frame - before < after - target_frame else after

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#frame extraction
async def extract_frame(video_path: str, frame_number: int, output_dir: str, index_path: str) -> str: #good
    """
    Asynchronously extract a specific frame from an MP4 video file using keyframe information.
    If the frame already exists in the cache, it returns the cached frame path.

    :param video_path: Path to the MP4 file
    :param frame_number: The frame number to extract
    :param output_dir: Directory to save extracted frames (png_cache)
    :param index_path: Path to the JSON index file
    :return: Path to the extracted or cached frame image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a standardized filename for the frame
    frame_filename = f"frame_{frame_number:08d}.png"
    frame_path = output_dir / frame_filename

    # Check if the frame already exists in the cache
    if frame_path.is_file():
        logger.info(f"Frame {frame_number} found in cache at {frame_path}")
        return str(frame_path)

    logger.info(f"Frame {frame_number} not found in cache, proceeding with extraction")

    indexer = AsyncVideoIndexer(video_path, index_path)
    await indexer.load_index()
    nearest_keyframe = indexer.find_nearest_keyframe(frame_number)

    logger.info(f"Seeking to nearest keyframe {nearest_keyframe} for target frame {frame_number}")

    try:
        with av_open(video_path) as container:
            stream = container.streams.video[0]
            
            # Calculate the timestamp for seeking
            time_base = stream.time_base
            target_ts = int(frame_number / (stream.average_rate * time_base))
            
            # Seek to the nearest keyframe
            container.seek(target_ts, stream=stream)
            
            # Decode frames
            for frame in container.decode(video=0):
                # Calculate current frame number
                current_frame_number = int(frame.pts * time_base * stream.average_rate)
                
                logger.debug(f"Processing frame: PTS={frame.pts}, Calculated frame number={current_frame_number}")
                
                if current_frame_number == frame_number:
                    img = frame.to_ndarray(format='rgb24')
                    cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    logger.info(f"Frame {frame_number} extracted and saved to {frame_path}")
                    return str(frame_path)
                elif current_frame_number > frame_number:
                    raise ValueError(f"Frame {frame_number} not found in video (overshot to {current_frame_number})")

        raise ValueError(f"Frame {frame_number} not found in video (reached end of stream)")

    except Exception as e:
        logger.error(f"Error extracting frame {frame_number}: {str(e)}")
        raise

async def main():
    pass