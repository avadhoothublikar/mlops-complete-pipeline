import pandas as pd
import os
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

#Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger('pre_processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "pre_processing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

def load_params(params_path: str) -> dict:
    """
    Load parameters from the YAML file
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

#string formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#Loading the tdidf processed data from processed folder
def load_data(file_path:str) -> pd.DataFrame:
    """
    Load data from csv file
    :param file_path: Path to the csv file
    :return Loaded DatFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded and %s with shape %s", file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s", e)
        raise
    except FileNotFoundError as e:
        logger.debug("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params:dict) -> RandomForestClassifier:
    """
    Train the RandamForest Model.
    :param x_train: Training features
    :param y_train: Training lebels
    :param params: Dictionay of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in x_train and y_train must me same.")
        
        logger.debug('Initializing RandomForest  model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug("Model training started with %d samples", x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug("Model training completed")
        return clf
    
    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
        raise
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to the file
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        #Ensure the directoty exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)

    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occured while saving the model: %s", e)
        raise


def main():
    try:
        #params = {'n_estimators':25, 'random_state':2}
        params = load_params(params_path='params.yaml')['model_training']
    

        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(x_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.debug("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()