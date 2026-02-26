import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import Porterstemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

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

    ps = Porterstemmer()
    #convert text to lower case
    text = text.lower()

    #Tokenize the text
    text = nltk.word_tokenize(text)

    #remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    #remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #stem the words
    text = [ps.stem(word) for word in text]

    #join the tokens back into single string
    return " ".join(text)


def preprocess_df(df, text_column='text', target_culumn='target'):
    """
    Preprocess the DataFrame by encoding the target column, removing duplicates and transforming the text column
    """

    try:
        logger.debug("starting preprocessing for DataFrame")
        