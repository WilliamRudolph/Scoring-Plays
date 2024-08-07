import logging
from pathlib import Path
from roboflow import Roboflow
import os


#LOGGING SETUP

def setup_logging(log_file: Path):

    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    except PermissionError:
        print(f"Permission denied when trying to create log file: {log_file}")
        print("Falling back to console-only logging.")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    print(logging.__file__)
    logger = logging.getLogger(__name__)
    return logger


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ROBOFLOW SETUP

def roboflow_setup(project_id: str, model_version: str, logger, confidence: float = 50.0):

    try:
        print("Setting up Roboflow API.")
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        print("Roboflow API key set.")
        project = rf.workspace().project(project_id)
        print("Roboflow project set.")
        model = project.version(model_version).model
        model.confidence = confidence
        print("Roboflow API setup complete.")
    except Exception as e:
        logger.error(f"Failure setting up Roboflow API: {e}")
        raise
    return model
