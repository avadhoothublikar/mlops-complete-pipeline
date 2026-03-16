import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk as nltk_import

nltk_import.download('stopwords')
nltk_import.download('punkt')

#Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#setup logger
logger = logging.getLogger('pre_processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "pre_processing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


#string formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation and stemming.
    """

    ps = PorterStemmer()
    #convert text to lower case
    text = text.lower()

    #Tokenize the text
    text = nltk_import.word_tokenize(text)

    #remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    #remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #stem the words
    text = [ps.stem(word) for word in text]

    #join the tokens back into single string
    return " ".join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocess the DataFrame by encoding the target column, removing duplicates and transforming the text column
    """

    try:
        logger.debug("starting preprocessing for DataFrame")
        #Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("target column is encoded")

        #Remove duplicate rows
        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicate rows are removed")

        #Apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    
    except KeyError as e:
        logger.error("column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s", e)


def main(text_column = 'text', target_column = 'target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """

    try:
        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        #Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(train_data, text_column, target_column)

        #store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"),index=False)

        logger.debug("Processed data saved to %s", data_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()