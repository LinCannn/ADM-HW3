from nltk import word_tokenize
from nltk.stem import PorterStemmer
import json
import pandas as pd

class SearchEngine:
    def __init__(self, original, df_path, vocabulary_path, inverted_index_path):
        self.original_df = pd.read_table(original)
        self.df = pd.read_table(df_path)
        temp_vocabulary = pd.read_csv(vocabulary_path)
        self.vocabulary = {word:index for word, index in zip(temp_vocabulary['word'], temp_vocabulary['term_id'])}
        with open(inverted_index_path) as f:
            self.inverted_index = json.load(f)
          


    def search(self,query):
        # Process the query terms
        query_text = word_tokenize(query.lower()) # Tokenize and lowercase query
        stemmer = PorterStemmer()
        query_text = [stemmer.stem(word) for word in query_text]
        term_ids = [] # List to store term IDs for query text
        # Check each query term in the vocabulary
        for term in query_text:
            if term in self.vocabulary: # Check if term is included in vocabulary
                term_ids.append(self.vocabulary[term])

        # Find restaurants containing all query terms
        if not term_ids:
            return [] # No terms matched in vocabulary
        
        ideal_restaurants = []
        # Narrow down results for each additional term
        for term_id in term_ids:
            term_id = str(term_id)
            ideal_restaurants += self.inverted_index[term_id]
        # Retrieve restaurant details from the DataFrame
        ideal_restaurants = set(ideal_restaurants)
        result = self.original_df[self.original_df['restaurantName'].isin(ideal_restaurants)]
        result = result[['restaurantName','address','description','website']]
        return result