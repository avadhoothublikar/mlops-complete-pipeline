import nltk_import
import os

# Force download into your venv's nltk_data directory
nltk_path = r'C:\Users\Avadhoot H\Documents\mlops\venv\nltk_data'
os.makedirs(nltk_path, exist_ok=True)

nltk_import.download('punkt_tab', download_dir=nltk_path)
nltk_import.download('punkt',     download_dir=nltk_path)
nltk_import.download('stopwords', download_dir=nltk_path)

# import nltk_import
# print(nltk_import.data.find('tokenizers/punkt_tab'))
# # Should print the path without error