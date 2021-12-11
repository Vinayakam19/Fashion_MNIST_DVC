from src.utils.common import read_yaml, create_directories
from src.utils.model import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import load_data, preprocess_data, one_hot_encode
import argparse
import os
import logging

STAGE = "04"

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])

    create_directories([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path  = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)
    
    data_dir = artifacts["DATA_DIR"]
    
    #Get the data directory path of the data folder
    data_dir_path = os.path.join(artifacts_dir, data_dir)

    #Load the data
    X_train, y_train = load_data(data_dir_path, kind='train')
    X_test, y_test = load_data(data_dir_path, kind='t10k')
    
    #preprocess the data and fit the model
    X_train, X_test = preprocess_data(X_train, X_test)
    
    #One-hot categorical encoding of the data
    y_train, y_test = one_hot_encode(y_train, y_test, params["CLASSES"])
    
    model.fit(
        X_train,y_train,
        validation_data=(X_test, y_test),
        batch_size=params["BATCH_SIZE"],
        epochs=params["EPOCHS"], 
        callbacks=callbacks
    )
    logging.info(f"training completed")

    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([trained_model_dir])

    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)
    logging.info(f"trained model is saved at: {model_file_path}")
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)