import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class CacheConfig:
    def __init__(self, max_age_days: int, max_size_gb: float, clean_frequency_hours: int):
        self.max_age_days = max_age_days
        self.max_size_gb = max_size_gb
        self.clean_frequency_hours = clean_frequency_hours

def clean_cache(cache_dir: Path, config: CacheConfig):
    config_file = cache_dir / "clean_config.json"
    
    # Check if it's time to clean
    if config_file.exists():
        with open(config_file, 'r') as f:
            last_clean = datetime.fromisoformat(json.load(f)['last_clean'])
        if datetime.now() - last_clean < timedelta(hours=config.clean_frequency_hours):
            logger.info(f"Skipping clean for {cache_dir}, last cleaned {last_clean}")
            return

    logger.info(f"Cleaning cache: {cache_dir}")

    if not cache_dir.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=config.max_age_days)
    files = [(f, f.stat().st_mtime, f.stat().st_size) for f in cache_dir.iterdir() if f.is_file()]
    files.sort(key=lambda x: x[1])
    
    total_size = sum(f[2] for f in files)
    max_size_bytes = config.max_size_gb * 1024 * 1024 * 1024

    for file_path, mtime, size in files:
        try:
            if datetime.fromtimestamp(mtime) < cutoff_date or total_size > max_size_bytes:
                os.remove(file_path)
                total_size -= size
                logger.info(f"Removed file: {file_path}")
            elif total_size <= max_size_bytes:
                break
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")

    # Update last clean time
    with open(config_file, 'w') as f:
        json.dump({'last_clean': datetime.now().isoformat()}, f)