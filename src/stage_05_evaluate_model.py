from src.utils.common import read_yaml, create_directories, save_reports, save_json
from src.utils.data_management import load_data
from keras.models import load_model
import argparse
import os
import logging
import numpy as np
import math
from sklearn.metrics import  accuracy_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import joblib

STAGE = "05"
    
def evaluate(config_path):
    config = read_yaml(config_path)
     
    artifacts = config['artifacts']
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["TRAINED_MODEL_DIR"])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    #model = joblib.load(model_path)
    model = load_model(model_path)
    
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    data_dir = artifacts["DATA_DIR"]
    
    #Get the data directory path of the data folder
    data_dir_path = os.path.join(artifacts_dir, data_dir)
    
    X_test, y_test = load_data(data_dir_path, "t10k")
    
    labels = y_test
    
    #prediction_by_class = model.predict_proba(X_test)
    predictions = np.argmax(model.predict(X_test), axis=1)
    #labels = np.argmax(y_test, axis=1)
    
    #PRC_json_path = config["plots"]["PRC"]
    #ROC_json_path = config["plots"]["ROC"]
    
    scores_json_path = config["metrics"]["SCORES"]
    
    #avg_prec = average_precision_score(labels, predictions)
    #roc_auc = roc_auc_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    
    scores = {
        #"avg_prec": avg_prec,
        #"roc_auc": roc_auc,
        "accuracy": accuracy
    }

    save_json(scores_json_path, scores)
    logging.info("Scores saved to: %s", scores_json_path)
    
    '''
    precision, recall, prc_threshold = precision_recall_curve(labels, predictions)
    nth_points = math.ceil(len(prc_threshold) / 1000)
    prc_points = list(zip(precision, recall, prc_threshold))[::nth_points]

    prc_data = {
        "prc": [
            {"precision": p, "recall": r, "threshold": t} 
            for p,r,t in prc_points
        ]
    }

    save_json(PRC_json_path, prc_data)
    '''
    '''
    fpr, tpr, roc_threshold = roc_curve(labels, predictions)

    roc_data = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold": t}
            for fp, tp, t in zip(fpr, tpr, roc_threshold)
        ]
    } 

    save_json(ROC_json_path, roc_data)
    '''
     
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        evaluate(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e