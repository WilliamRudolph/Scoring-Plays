# IMPORTS

import asyncio
import aiohttp
import aiofiles
from typing import List, Optional, Callable, Tuple, Union, Dict
from pathlib import Path
import roboflow
import os
from dotenv import load_dotenv
import tempfile
import hashlib
from cache_clean import clean_cache, CacheConfig
import numpy as np
import json
from pprint import pprint as pp
import csv
# Create a temporary directory for matplotlib
mpl_tempdir = tempfile.mkdtemp()
os.environ['MPLCONFIGDIR'] = mpl_tempdir
import subprocess
from setup import setup_logging, roboflow_setup

from frame_extraction import AsyncVideoIndexer, extract_frame

#--------------------------------------------------------------------------------------------------------------
#ROBOFLOW SETUP

# we need to edit this to use the roboflow_setup function from setup.py
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_ID = 'in-game'
MODEL_VERSION = '2'

rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_ID)
model = project.version(MODEL_VERSION).model

#--------------------------------------------------------------------------------------------------------------
#LOGGING AND CACHE SETUP

fps_cache: Dict[str, float] = {}

async def get_video_fps(video_path: str) -> float:
    """
    Asynchronously get the frames per second (fps) of an MP4 video file.

    :param video_path: Path to the MP4 file
    :return: Frames per second as a float
    """
    video_path = str(Path(video_path).resolve())  # Normalize path

    # Check if we've already processed this file
    if video_path in fps_cache:
        logger.info(f"Using cached fps value for {video_path}")
        return fps_cache[video_path]

    try:
        # Construct the FFprobe command
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=r_frame_rate,nb_read_packets",
            "-of", "json",
            video_path
        ]

        # Run the FFprobe command asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for the subprocess to complete and capture output
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr.decode())

        # Parse the JSON output
        data = json.loads(stdout.decode())
        
        # Extract the frame rate fraction
        r_frame_rate = data['streams'][0]['r_frame_rate']
        
        # Convert the fraction to a float
        numerator, denominator = map(int, r_frame_rate.split('/'))
        fps = numerator / denominator

        # Cache the result
        fps_cache[video_path] = fps

        logger.info(f"Successfully extracted fps for {video_path}: {fps}")
        return fps

    except asyncio.SubprocessError as e:
        logger.error(f"Error running FFprobe: {e}")
        raise

    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        logger.error(f"Error parsing FFprobe output for {video_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error getting fps for {video_path}: {e}")
        raise

def get_video_specific_cache_dir(base_dir: Path, m3u8_url: str) -> Path:
    # Create a unique identifier for the video based on its URL
    video_id = hashlib.md5(m3u8_url.encode()).hexdigest()
    return base_dir / video_id

#set up the log file
log_dir = Path.home() / 'Documents' / 'basket_detection_cache'
log_file = log_dir / 'video_processing.log'

logger = setup_logging(log_file)



#--------------------------------------------------------------------------------------------------------------
# FRAME ANALYSIS

async def classify_frame(frame_path: str, target_class: str = 'in-game', confidence_threshold: float = 0.8, pprint: bool = False): #good
    """
    Classify a frame using Roboflow API call.

    :param frame_path: Path to the frame.
    :param confidence_threshold: Threshold of model confidence to filter out bad predictions. Default is 0.8.
    :param pprint: Bool whether we should print the Roboflow results in JSON format. Default to false.

    """
    try:
        response = model.predict(frame_path).json()
        if pprint:
            pp(response)
    except Exception as e:
        logger.error(f"Error classifying frame: {e}")
        return None

    predictions = response['predictions'][0]
    confidence = predictions.get('confidence')
    top_class = predictions.get('top')

    if confidence is None:
        logger.error(f"Could not find confidence for {frame_path}")
        return None
    elif confidence < confidence_threshold:
        logger.error(f"Confidence of {confidence} below threshold {confidence_threshold}")
        return None
    if top_class == target_class:
        return True
    if top_class != target_class:
        return False
    

#--------------------------------------------------------------------------------------------------------------
# VIDEO ANALYSIS AND TRANSITION DETECTION

async def detect_game_states(video_path: str, index_path: str, png_cache_dir: str, fps: float, initial_state_sample_duration: float = 5.0) -> List[Tuple[float, float, str]]:
    """
    Detect game states (in-game/out-game) throughout the video.

    :param video_path: Path to the video file
    :param index_path: Path to the video index file
    :param png_cache_dir: Directory to save extracted frames (our png cache)
    :param fps: Frames per second of the video
    :param initial_state_sample_duration: Duration (in seconds) to sample for initial state determination
    :return: List of tuples (start_time, end_time, state)
    """
    #initial_state, confidence, *transition = await determine_initial_state(video_path, index_path, png_cache_dir, fps, initial_state_sample_duration)
    initial_state, confidence, transition = True, 1.00, None
    
    indexer = AsyncVideoIndexer(video_path, index_path)
    await indexer.load_index()
    total_frames = indexer.index['total_frames']
    duration = total_frames / fps

    coarse_interval = 600  # seconds (20 is optimal)
    fine_interval = 100  # seconds (2 is optimal)

    state_changes = []
    current_state = initial_state
    last_change_time = 0

    # If a transition was detected in the initial state determination, add it
    if transition:
        transition_time = transition[0]
        state_changes.append((0, transition_time, "out-game" if initial_state else "in-game"))
        last_change_time = transition_time
        current_state = not initial_state
        start_time = last_change_time
    else:
        # If no early transition, add the initial state
        state_changes.append((0, initial_state_sample_duration, "in-game" if initial_state else "out-game"))
        start_time = last_change_time = initial_state_sample_duration

    # Coarse pass (starting from the end of initial state analysis)
    
    async with aiohttp.ClientSession() as session:
        for time in range(int(start_time), int(duration), coarse_interval):
            frame_number = int(time * fps)
            frame_path = str(await extract_frame(video_path, frame_number, png_cache_dir, index_path))
            print(f"Coarse pass frame path for frame number {frame_number}: {frame_path}")
            print(f"Frame path var type: {type(frame_path)}")
            pass
            is_in_game = await classify_frame(frame_path)
            print(f"Is in-game: {is_in_game}")

            if is_in_game is None:
                logger.warning(f"Uncertain classification for frame at {time}s, skipping")
                continue

            if current_state != is_in_game:
                # Potential state change detected, perform fine-grained analysis
                exact_change_time = await fine_grained_analysis(
                    video_path, index_path, png_cache_dir, fps, session,
                    last_change_time, time, current_state, is_in_game, fine_interval
                )
                state_changes.append((last_change_time, exact_change_time, "in-game" if current_state else "out-game"))
                last_change_time = exact_change_time
                current_state = is_in_game

    # Add the final state
    state_changes.append((last_change_time, duration, "in-game" if current_state else "out-game"))

    return state_changes

# Update the determine_initial_state function to accept sample_duration as a parameter
async def determine_initial_state(video_path: str, index_path: str, output_dir: str, fps: float, sample_duration: float = 5.0) -> Tuple[bool, float, Optional[float]]: #good
    """
    Determine the initial state of the game by analyzing the first few seconds of the video.
    
    :param video_path: Path to the video file
    :param index_path: Path to the video index file
    :param output_dir: Directory to save extracted frames
    :param fps: Frames per second of the video
    :param sample_duration: Duration (in seconds) to sample for initial state determination
    :return: Tuple of (is_in_game: bool, confidence: float, transition_time: Optional[float])
    """
    sample_interval = 0.5  # Sample every 0.5 seconds
    
    sample_times = np.arange(0, sample_duration, sample_interval)
    states = []


    try:
        for time in sample_times:
            frame_number = int(time * fps) + 1
            frame_path = await extract_frame(video_path, frame_number, output_dir, index_path)
            is_in_game = await classify_frame(frame_path)
            if is_in_game is not None:
                states.append(is_in_game)

    except Exception as e:
        print(f"Initial state classification fail: {e}")

    # Calculate the dominant state and confidence
    if not states:
        logger.warning("No valid classifications during initial state determination")
        return False, 0.0, None

    in_game_count = sum(states)
    total_samples = len(states)
    is_in_game = in_game_count > total_samples / 2
    confidence = max(in_game_count, total_samples - in_game_count) / total_samples

    logger.info(f"Initial state determined as {'in-game' if is_in_game else 'out-game'} with {confidence:.2f} confidence")

    # Check for potential quick state change
    if 0 < in_game_count < total_samples:
        transition_point = next((i for i, (a, b) in enumerate(zip(states, states[1:])) if a != b), None)
        if transition_point is not None:
            transition_time = sample_times[transition_point]
            logger.info(f"Potential state change detected at {transition_time:.2f} seconds")
            return is_in_game, confidence, transition_time

    return is_in_game, confidence, None

async def fine_grained_analysis(video_path: str, index_path: str, output_dir: str, fps: float,
                                session: aiohttp.ClientSession, start_time: float, end_time: float,
                                start_state: bool, end_state: bool, interval: float) -> float:
    for time in np.arange(start_time, end_time, interval):
        frame_number = int(time * fps)
        frame_path = await extract_frame(video_path, frame_number, output_dir, index_path)
        is_in_game = await classify_frame(session, frame_path)

        if is_in_game is None:
            continue  # Skip uncertain classifications

        if is_in_game != start_state:
            # Confirm the state change
            if await confirm_state_change(video_path, index_path, output_dir, fps, session, time, is_in_game):
                return time

    # If no clear change point is found, return the midpoint
    return (start_time + end_time) / 2

async def confirm_state_change(video_path: str, index_path: str, output_dir: str, fps: float,
                               session: aiohttp.ClientSession, change_time: float, new_state: bool,
                               confirmation_frames: int = 5) -> bool:
    for offset in range(1, confirmation_frames + 1):
        frame_number = int((change_time + offset * 0.5) * fps)  # Check half a second apart
        frame_path = await extract_frame(video_path, frame_number, output_dir, index_path)
        is_in_game = await classify_frame(session, frame_path)
        if is_in_game is None or is_in_game != new_state:
            return False
    return True
#--------------------------------------------------------------------------------------------------------------

async def save_transitions_to_file(transitions: List[Tuple[float, float, str]], base_dir: Path, filename: str = 'game_transitions.csv'):
    """
    Save the detected transitions to a CSV file in the specified base directory.
    
    :param transitions: List of tuples (start_time, end_time, state) as returned by detect_game_states
    :param base_dir: Base directory to save the file in
    :param filename: Name of the file to save the transitions (default: 'game_transitions.csv')
    """
    filepath = base_dir / filename
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with aiofiles.open(filepath, mode='w', newline='') as afp:
            writer = csv.writer(afp)
            await writer.writerow(['Start Time (s)', 'End Time (s)', 'Duration (s)', 'State'])  # Header

            for i, (start_time, end_time, state) in enumerate(transitions):
                duration = end_time - start_time
                await writer.writerow([f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}", state])

                if i < len(transitions) - 1:
                    next_start, _, next_state = transitions[i+1]
                    if next_start != end_time:
                        logger.warning(f"Gap or overlap detected between segments at {end_time:.2f}s")

        logger.info(f"Transitions saved to {filepath.absolute()}")
    except Exception as e:
        logger.error(f"Error saving transitions to file: {e}")
        raise

#--------------------------------------------------------------------------------------------------------------

async def main():

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
    logger.info(f"Cache directories set up: {base_cache_dir}")

    fps = await get_video_fps(video_path)
    print(f"Video name: {video_name}")
    print(f"Path: {video_path}")
    print(f"Index path: {index_path}")
    #print(await determine_initial_state(video_path, index_path, png_cache_dir, fps))

    await detect_game_states(video_path, index_path, png_cache_dir, fps)

if __name__ == '__main__':
    asyncio.run(main())
    