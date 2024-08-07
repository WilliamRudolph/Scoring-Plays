from operator import is_
from re import split
from venv import create
from aiohttp import UnixConnector
from click import confirm
import numpy as np
import cv2
from typing import List, Tuple, Dict, Union
from pathlib import Path
import csv
from os import path, mkdir, listdir
import asyncio
import uuid
import os
import pprint
import sys


from setup import setup_logging, roboflow_setup
from frame_extraction import extract_frame
from game_state_detection import get_video_fps


#--------------------------------------------------------------------------------------------------------------
#ROBOFLOW SETUP, GAME STATE TRANSITION, PATH IMPORTS, LOGGING SETUP

from params import video_path, base_cache_dir, png_cache_dir, index_path

log_dir = Path.home() / 'Documents' / 'basket_detection_cache'
log_file = log_dir / 'video_processing.log'

logger = setup_logging(log_file)

model = roboflow_setup('baskets-made', '4', logger)
print(f"Roboflow model: {model}")


# Define the game state transition times

# game_state_transition = [
#     (0.00, 280.00, "out-game"), #-> pre-game
#     (280.00, 2504, "in-game"),  #-> first half
#     (2504, 3117, "out-game"),   #-> half time
#     (3117, 5391, "in-game"),    #-> second half
#     (5391, 5498, "out-game"),   #-> break before over time
#     (5498, 6515, "in-game"),    #-> over time
#     (6515, 7256, "out-game")   #-> post-game
# ]

def seconds_to_timecodes(tuples_list):
    def seconds_to_timecode(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    return [(seconds_to_timecode(t[0]), seconds_to_timecode(t[1]), t[2]) for t in tuples_list]

def seconds_to_frames(tuples_list):
    def seconds_to_frame(seconds):
        return int(seconds * 25)

    return [(seconds_to_frame(t[0]), seconds_to_frame(t[1]), t[2]) for t in tuples_list]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GAME STATE PROCESSING

async def find_intervals(transition_list: List[Tuple[float, float, str]], target_class: str = "in-game") -> List[Tuple[float, float]]:
    relevant_intervals = []
    try:
        for start_time, end_time, state in transition_list:
            if state == target_class:
                relevant_intervals.append((start_time, end_time))
    except Exception as e:
        logger.error(f"Failure finding relevant intervals: {e}")
    return relevant_intervals

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FRAME PROCESSING

async def create_top_half_mosaic(
    video_path: str,
    frame1_number: int,
    frame2_number: int,
    output_dir: Path,
    cache_dir: Path,
    index_path: str,
    target_width: int = 1920
) -> str:
    """
    Create a mosaic from the top halves of two frames, handling various sizes and color spaces.
    
    :param video_path: Path to the video file
    :param frame1_number: Frame number of the first frame
    :param frame2_number: Frame number of the second frame
    :param output_dir: Directory to save the output mosaic
    :param cache_dir: Directory for frame extraction cache
    :param index_path: Path to the video index file
    :param target_width: Target width for the mosaic (default is 1920)
    :return: Path to the saved mosaic image
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames concurrently
        frame1_path, frame2_path = await asyncio.gather(
            extract_frame(video_path, frame1_number, str(cache_dir), index_path),
            extract_frame(video_path, frame2_number, str(cache_dir), index_path)
        )

        # Generate output filename
        output_filename = f"mosaic_f{frame1_number:08d}_f{frame2_number:08d}.png"
        output_path = output_dir / output_filename

        if output_path.is_file():
            logger.info(f"Mosaic already exists for frames {frame1_number} and {frame2_number}.")
            return str(output_path)

        # Read frames
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        # Process frames
        top_half1 = process_frame(frame1, target_width)
        top_half2 = process_frame(frame2, target_width)

        # Stack the top halves vertically
        mosaic = np.vstack((top_half1, top_half2))

        # Ensure the final height is even (for some codecs)
        if mosaic.shape[0] % 2 != 0:
            mosaic = mosaic[:-1, :, :]

        

        # Save the mosaic
        cv2.imwrite(str(output_path), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
        logger.info(f"Mosaic saved for frames {frame1_number} and {frame2_number}.")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error creating mosaic for frames {frame1_number} and {frame2_number}: {e}")
        raise

def process_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Process a frame: convert color space, resize if necessary, and extract top half.
    """
    # Convert to RGB if it's not
    if len(frame.shape) == 2:  # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif frame.shape[2] == 3 and frame.dtype == np.uint8:
        # Assume it's BGR if it's uint8 with 3 channels
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize if necessary
    if frame.shape[1] != target_width:
        aspect_ratio = frame.shape[0] / frame.shape[1]
        target_height = int(target_width * aspect_ratio)
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Extract top half
    return frame[:frame.shape[0]//2, :, :]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BASKET DETECTION

async def process_single_frame(frame_path: path) -> bool:
    """
    Process a single frame to detect made baskets.

    :param frame_path: Path to the frame image file
    :return: Boolean value indicating made basket detection
    """
    try:
        # Run the inference
        result = model.predict(frame_path, confidence=40, overlap=30).json()

        # Process predictions
        for prediction in result['predictions']:
                if prediction['class'] == 'made-basket':
                    return True
                return False
    except Exception as e:
        logger.error(f"Error processing frame {frame_path}: {str(e)}")
        return False

async def process_mosaicked_frame(mosaic_path: str, frame1_number: int, frame2_number: Union[int, None] = None) -> Dict[int, bool]:
    """
    Process a mosaicked frame or single frame to detect made baskets.

    :param mosaic_path: Path to the mosaicked image file or single frame
    :param frame1_number: The frame number of the first (or only) frame
    :param frame2_number: The frame number of the second frame (if applicable), or None for single frame
    :return: A dictionary with frame numbers as keys and boolean values indicating made basket detection
    """
    try:
        # Run the inference
        result = model.predict(mosaic_path, confidence=50, overlap=30).json()

        # Extract image height
        image_height = int(result['image']['height'])

        # Determine if we're processing a single frame or a mosaic
        is_mosaic = frame2_number is not None

        if is_mosaic:
            midpoint = image_height // 2
            frame1_made_basket = False
            frame2_made_basket = False

            # Process predictions for mosaic
            for prediction in result['predictions']:
                if prediction['class'] == 'made-basket':
                    if prediction['y'] < midpoint:
                        frame1_made_basket = True
                    else:
                        frame2_made_basket = True

            return {
                frame1_number: frame1_made_basket,
                frame2_number: frame2_made_basket
            }
        else:
            # Process predictions for single frame
            frame_made_basket = any(pred['class'] == 'made-basket' for pred in result['predictions'])
            return {frame1_number: frame_made_basket}

    except Exception as e:
        logger.error(f"Error processing frame(s) {mosaic_path}: {str(e)}")
        # In case of an error, return False for all frames
        return {frame1_number: False} if frame2_number is None else {frame1_number: False, frame2_number: False}

async def detect_made_baskets(
    video_path: str,
    game_state_transitions: List[Tuple[float, float, str]],
    cache_dir: Path,
    split_cache_dir: Path,
    frame_sample_rate: int = 4,
    index_path: str = None,
    fps: int = 25,
    max_concurrent_tasks: int = 20,
    chunk_size: int = 100,
    save: bool = False
) -> List[int]:
    """
    Detect made baskets in the given game intervals.

    :param video_path: Path to the video file
    :param game_state_transitions: List of tuples (start_time, end_time, state)
    :param frame_sample_rate: Number of frames to skip between samples
    :param cache_dir: Directory for caching extracted frames
    :param index_path: Path to the video index file
    :param fps: Frames per second of the video
    :param save: Whether to save the list of made basket frame numbers as a txt file
    :return: List of frame numbers where made baskets were detected
    """
    # Start out with a set to avoid duplicates
    preliminary_made_basket_frames = set()
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def process_frames(frame1: int, frame2: int) -> Dict[int, bool]:
        async with semaphore:
            mosaic_path = await create_top_half_mosaic(
                video_path, frame1, frame2, split_cache_dir, cache_dir, index_path
            )
            return await process_mosaicked_frame(mosaic_path, frame1, frame2)

    async def process_chunk(frames: List[Tuple[int, int]]) -> List[Dict[int, bool]]:
        tasks = [process_frames(f1, f2) for f1, f2 in frames]
        return await asyncio.gather(*tasks)
    
    for start_time, end_time, state in game_state_transitions:
        if state != "in-game":
            continue

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        frame_pairs = []
        for frame in range(start_frame, end_frame, frame_sample_rate):
            next_frame = min(frame + frame_sample_rate, end_frame)
            if frame != next_frame:
                frame_pairs.append((frame, next_frame))
            else:
                frame_pairs.append((frame, None))


        # Process frame pairs in chunks
        for i in range(0, len(frame_pairs), chunk_size):
            chunk = frame_pairs[i:i + chunk_size]
            results = await process_chunk(chunk)
            for result in results:
                for frame, is_made in result.items():
                    if is_made:
                        preliminary_made_basket_frames.add(frame)
    # Convert set to list
    prelim_made_basket_list = list(preliminary_made_basket_frames)

    if save:
        try:
            with open(cache_dir / "made_basket_frames.txt", "w") as f:
                f.write("\n".join(map(str, sorted(prelim_made_basket_list))))
                logger.info("Made basket frames saved to made_basket_frames.txt.")
        except Exception as e:
            logger.error(f"Error saving made basket frames: {e}")

    return sorted(prelim_made_basket_list)

async def confirm_basket_detection(made_basket_frames: List[int], video_path: str, cache_dir: Path, index_path: str) -> List[int]:
    """
    :param made_basket_frames: List of frame numbers where made baskets were detected
    :param video_path: Path to the video file
    :param cache_dir: Directory for caching extracted frames
    :param index_path: Path to the video index file
    :return: List of confirmed frame numbers where made baskets
    """
    confirmed = []
    unconfirmed = []

    # First, we can confirm any made basket frame where the subsequent frame is also a made basket frame. 
    # (If frame 1, 5, 9 are made baskets, we can confirm 1.)

    i = 0
    while i < len(made_basket_frames):
        current = made_basket_frames[i]
        j = i + 1
        found_sequence = False

        while j < len(made_basket_frames) and made_basket_frames[j] == made_basket_frames[j - 1] + 4:
            found_sequence = True
            j += 1

        if found_sequence:
            confirmed.append(current)
            print(f"{current} confirmed.")
        else:
            unconfirmed.append(current)
            print(f"{current} unconfirmed.")

        i = j
    if not unconfirmed:
        return sorted(confirmed)
    print("Moving to second stage of confirmation.")
    # With the remaining baskets, we check the frames before and after to see if they are also made baskets. 
    # If they are, we can confirm the middle frame.

    for frame in unconfirmed:
        detected = []
        undetected = []
        for offset in range(-3, 4):
            if offset != 0:
                check_frame_num = frame + offset
                if await process_single_frame(await extract_frame(video_path, check_frame_num, str(cache_dir), index_path)):
                    detected.append(check_frame_num)
                    print(f"Frame {check_frame_num} confirmed.")
                else:
                    undetected.append(check_frame_num)
                    print(f"Frame {check_frame_num} unconfirmed.")
        if len(detected) >=2:
            confirmed.append(frame)
    
    # Remove unconfirmed frames from the list
    unconfirmed = [frame for frame in unconfirmed if frame not in confirmed]
    # Log the unconfirmed frames
    if unconfirmed:
        logger.info(f"Frames {unconfirmed} failed confirmation.")
    else:
        logger.info("All frames confirmed.")
    
    return sorted(confirmed)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FINAL BASKET DETECTION MAIN FUNCTION

async def basket_detection_main(
    video_path: str,
    game_state_transitions: List[Tuple[float, float, str]],
    cache_dir: Path,
    split_cache_dir: Path,
    fps: int = 25,
):
    
    relevant_intervals = await find_intervals(game_state_transitions, "in-game")
    

    prelim = await detect_made_baskets(video_path, relevant_intervals, cache_dir, split_cache_dir, fps=fps, save=True)
    final = await confirm_basket_detection(prelim, video_path, cache_dir, index_path)
    return final

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
async def main():
    # Define the video path
    video_name = 'gam3974357c68.mp4'
    video_path = Path.home() / 'rps' / video_name
    

    # Define the base cache directory
    base_cache_dir = Path.home() / 'Documents' / 'basket_detection_cache'

    # Define cache directories
    base_cache_dir = Path.home() / 'Documents' / 'basket_detection_cache'
    png_cache_dir = base_cache_dir / 'png_files'
    split_cache_dir = base_cache_dir / 'split_frames'
    index_path = base_cache_dir / f"{os.path.splitext(video_name)[0]}_index_cv2.json"

    # Create cache directories if they don't exist
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    png_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directories set up: {base_cache_dir}")       

if __name__ == '__main__':

    asyncio.run(main())
