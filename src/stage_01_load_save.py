import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
import requests

STAGE = "01" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def download_fmnist(path, source_url):
    DEFAULT_SOURCE_URL = source_url
    files = dict(
        TRAIN_IMAGES='train-images-idx3-ubyte.gz',
        TRAIN_LABELS='train-labels-idx1-ubyte.gz',
        TEST_IMAGES='t10k-images-idx3-ubyte.gz',
        TEST_LABELS='t10k-labels-idx1-ubyte.gz')
    if not os.path.exists(path):
        os.mkdir(path)
    for f in files:
        filepath = os.path.join(path, files[f])
        if not os.path.exists(filepath):
            url = DEFAULT_SOURCE_URL + files[f]
            r = requests.get(url, allow_redirects=True)
            open(filepath, 'wb').write(r.content)

def get_data(config_path):
    config = read_yaml(config_path)
    
    #Read the remote url(data source) from the config file
    remote_data_path = config['data_source']
    
    #Save the data to a local directory under the path
    #The downloaded data will be saved to the local directory: artifacts/data/
    artifacts_dir = config['artifacts']['ARTIFACTS_DIR']
    data_dir = config['artifacts']['DATA_DIR']

    #Get the directory path of the data folder
    data_dir_path = os.path.join(artifacts_dir, data_dir)
    
    #Create the data directory
    create_directories([data_dir_path])
    
    #Download the fashion mnist data from the remote url
    download_fmnist(data_dir_path, remote_data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        get_data(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e