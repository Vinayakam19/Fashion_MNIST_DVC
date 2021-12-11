from src.utils.common import read_yaml, create_directories
from src.utils.callbacks import create_and_save_tensorboard_callback, create_and_save_checkpoint_callback
import argparse
import os
import logging

STAGE = "03"

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def prepare_callbacks(config_path):
    config = read_yaml(config_path)
   
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    
    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts['TENSORBOARD_ROOT_LOG_DIR'])
    
    checkpoint_dir = os.path.join(artifacts_dir, artifacts['CHECKPOINT_DIR'])
    
    callbacks_dir = os.path.join(artifacts_dir, artifacts['CALLBACKS_DIR'])
    
    create_directories([tensorboard_log_dir, checkpoint_dir, callbacks_dir])
    
    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_callbacks(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e