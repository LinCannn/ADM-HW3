import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
import pandas

class Preprocessor():
    """
    A class for preprocessing text data, which includes cleaning text,
    removing stopwords and punctuation, and applying stemming.

    Methods:
    - __init__: Initializes the necessary components for text preprocessing.
    - text_cleaner: Cleans and preprocesses a given text.
    - filter: Applies text cleaning to a specific column in a DataFrame.
    """
    
    def __init__(self):
        """
        Initializes the Preprocessor class by downloading necessary NLTK data
        and creating a PorterStemmer instance for stemming words.
        """
        # Download the list of English stopwords
        nltk.download("stopwords")
        # Download the Punkt tokenizer models
        nltk.download("punkt_tab")  
        # Initialize the Porter stemmer for stemming words
        self.stemmer = PorterStemmer()
    
    def text_cleaner(self, text):
        """
        Cleans and preprocesses the input text by removing stopwords and punctuation,
        and applying stemming.

        Args:
        - text (str): The input text to be cleaned.

        Returns:
        - str: The cleaned and preprocessed text as a single string.
        """
        # Define the set of English stopwords
        stop_words = set(stopwords.words("english"))
        # Define the set of punctuation characters
        punctuation = set(string.punctuation)

        # Tokenize the input text into words
        words = word_tokenize(text)

        # Remove stopwords and punctuation, then apply stemming
        filtered_words = [
            self.stemmer.stem(word) for word in words
            if word.lower() not in stop_words and word not in punctuation
        ]

        # Combine the filtered words back into a single string
        return ' '.join(filtered_words)
    
    def filter(self, df):
        """
        Applies text cleaning to the "description" column of a DataFrame
        and creates a new column "description_filtered" with the cleaned text.

        Args:
        - df (pandas.DataFrame): The input DataFrame containing a "description" column.

        Returns:
        - pandas.DataFrame: The modified DataFrame with an added "description_filtered" column.
        """
        # Apply text cleaning to the "description" column and store in "description_filtered"
        df["description_filtered"] = df["description"].str.lower().apply(self.text_cleaner)
        return df
