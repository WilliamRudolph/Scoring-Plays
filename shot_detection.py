from pathlib import Path
import asyncio
from typing import List, Dict, Tuple
from roboflow import Roboflow
from setup import setup_logging, roboflow_setup
from params import video_path, base_cache_dir, png_cache_dir, index_path
from frame_extraction import extract_frame
from pprint import pprint

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#LOGGING AND MODEL SETUP

log_dir = Path.home() / 'Documents' / 'basket_detection_cache'
log_file = log_dir / 'video_processing.log'

logger = setup_logging(log_file)

model = roboflow_setup('shot-taken', '3', logger)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

made_basket_frames = []

async def detect_scoring_sequences(
    video_path: str,
    made_basket_frames: List[int],
    look_back_frames: int = 15,  # Assuming 25 fps, this looks back ~0.6 seconds
    cache_dir: Path = Path(png_cache_dir),
    index_path: str = index_path,
    max_concurrent_tasks: int = 20
) -> List[Dict[str, int]]:
    """
    Detect scoring sequences by looking for shooting players before made baskets.

    :param video_path: Path to the video file
    :param made_basket_frames: List of frame numbers where made baskets were detected
    :param shooting_player_model: Object detection model for detecting shooting players
    :param look_back_frames: Number of frames to look back for a shooting player
    :param cache_dir: Directory for caching extracted frames
    :param index_path: Path to the video index file
    :param max_concurrent_tasks: Maximum number of tasks to run concurrently
    :return: List of dictionaries containing made basket and shooting player frame numbers
    """
    scoring_sequences = []
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def detect_shooting_player(frame_number: int) -> Tuple[int, bool]:
        async with semaphore:
            frame_path = await extract_frame(video_path, frame_number, str(cache_dir), index_path)
            result = await model.predict(frame_path, confidence=30, overlap=30).json()
            has_shooting_player = any(pred['class'] == 'shooting-player' for pred in result['predictions'])
            return frame_number, has_shooting_player

    for made_basket_frame in made_basket_frames:
        start_frame = max(0, made_basket_frame - look_back_frames)
        frames_to_check = range(start_frame, made_basket_frame)

        # Check frames for shooting player
        tasks = [detect_shooting_player(frame) for frame in frames_to_check]
        results = await asyncio.gather(*tasks)

        # Find the latest frame with a shooting player
        shooting_player_frames = [frame for frame, has_player in results if has_player]
        if shooting_player_frames:
            shooting_player_frame = max(shooting_player_frames)
            scoring_sequences.append({
                'made_basket_frame': made_basket_frame,
                'shooting_player_frame': shooting_player_frame
            })
        else:
            # If no shooting player found, still record the made basket
            scoring_sequences.append({
                'made_basket_frame': made_basket_frame,
                'shooting_player_frame': None
            })

    return scoring_sequences


async def main():
    for frame_number in range(8812, 8826):
        frame_path = await extract_frame(video_path, frame_number, str(png_cache_dir), index_path)
        result = model.predict(frame_path, confidence=30, overlap=30).json()
        print(f"Frame {frame_number}:")
        pprint(result)

if __name__ == '__main__':
    asyncio.run(main())