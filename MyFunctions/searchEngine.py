from nltk import word_tokenize
from nltk.stem import PorterStemmer
import json
import pandas as pd
import math
import numpy as np

class SearchEngine:
    def __init__(self, df_path, vocabulary_path, inverted_index_path):
        self.df = pd.read_table(df_path)
        temp_vocabulary = pd.read_csv(vocabulary_path)
        self.vocabulary = {word: index for word, index in zip(temp_vocabulary['word'], temp_vocabulary['term_id'])}
        with open(inverted_index_path) as f:
            self.inverted_index = json.load(f)
          
    def search(self, query):
        # Process the query terms
        query_text = word_tokenize(query.lower())  # Tokenize and lowercase query
        stemmer = PorterStemmer()
        query_text = [stemmer.stem(word) for word in query_text]
        
        term_ids = [self.vocabulary[term] for term in query_text if term in self.vocabulary]
        
        if not term_ids:
            return []  # No terms matched in vocabulary
        
        # Narrow down results for each additional term
        ideal_restaurants = set()
        for term_id in term_ids:
            ideal_restaurants.update(self.inverted_index[str(term_id)])
        
        # Retrieve restaurant details from the DataFrame
        result = self.df[self.df['restaurantName'].isin(ideal_restaurants)]
        return result[['restaurantName', 'address', 'description', 'website']]
        
    def tf(self, description, word_index):
        tf_res = np.zeros(len(word_index))
        for word in word_index.keys():
            tf_res[word_index[word]] = description.count(word) / len(description)
        return tf_res
    
    def idf(self, word_index, total_documents):
        vocabulary_index = pd.read_csv("dataset/vocabulary.csv")
        indexes = []
        idf_res = np.zeros(len(word_index))
        for word in word_index.keys():
            index = vocabulary_index[vocabulary_index['word'] == word]['term_id']
            indexes.append(int(index))
        
        with open("dataset/inverted_index.json","r") as f:
            word_dict = json.load(f)
            for pos, i in enumerate(indexes):
                idf =  math.log(total_documents / (1 + len(word_dict[str(i)])))
                idf_res[pos] = idf
        return idf_res
    
    def tf_idf_score(self, tf, idf): # Hadamart product
        tf_idf = tf * idf  
        return tf_idf
    

    def compute_cosine_similarity(self, v, w): #given 2 vectors v and w
        # Ensure inputs are numpy arrays
        v = np.array(v, dtype=float)
        w = np.array(w, dtype=float)
        
        # Handle zero vectors to avoid division by zero
        if np.all(v == 0) or np.all(w == 0):
            return 0.0
        
        # Normalize vectors
        v = v / (np.sum(v) if np.sum(v) != 0 else 1)
        w = w / (np.sum(w) if np.sum(w) != 0 else 1)
        
        # Calculate magnitudes
        magnitude_v = np.sqrt(np.sum(v**2))
        magnitude_w = np.sqrt(np.sum(w**2))
        
        # Compute dot product and cosine similarity
        dot_product = np.sum(v * w)
        
        # Return a single scalar value
        return float(dot_product / (magnitude_v * magnitude_w))

    def get_restaurant_scores(self, query, k=5):
        # Process query
        query_tokens = word_tokenize(query.lower())
        stemmer = PorterStemmer()
        word_index = {stemmer.stem(word): i for i, word in enumerate(query_tokens)}
        
        # Calculate query vectors
        query_tf = self.tf(query, word_index)
        tot_doc = self.df.shape[0]
        query_idf = self.idf(word_index, tot_doc)
        query_tf_idf = self.tf_idf_score(query_tf, query_idf)
        
        # Get relevant restaurants
        df_temp = self.search(query)
        df_results = self.df[self.df['restaurantName'].isin(df_temp['restaurantName'])].copy()
        
        if df_results.empty:
            return df_results
        
        # Calculate TF-IDF scores
        descriptions = df_results['description_filtered'].tolist()
        description_tokens = [description.split(' ') for description in descriptions]
        doc_tf = np.array([self.tf(tokens, word_index) for tokens in description_tokens])
        doc_tf_idf = self.tf_idf_score(doc_tf, query_idf)
        
        # Calculate cosine similarities
        cos_sim = []
        for doc_vector in doc_tf_idf:
            similarity = self.compute_cosine_similarity(doc_vector, query_tf_idf)
            cos_sim.append(similarity)
        
        # Add columns directly to the dataframe
        df_results['tf_idf'] = [vector.tolist() for vector in doc_tf_idf]
        df_results['similarityScore'] = cos_sim
        df_results.sort_values(by='similarityScore', ascending=False, inplace=True)
        # Sort and return top k results
        return df_results[['restaurantName','address','description','website','similarityScore']].head(k)



class AdvancedSearchEngine:
    def __init__(self):
        # Load the original dataset
        self.df = pd.read_table("dataset/restaurant_info.tsv")
    
    def get_price_range(self, min_price, max_price):
        """
        Determines the price range list based on the minimum and maximum price symbols provided.
        
        Args:
            min_price (str): Minimum price symbol, e.g., '€'.
            max_price (str): Maximum price symbol, e.g., '€€€€'.
            
        Returns:
            List of price symbols in the specified range.
        """
        prices = ['€', '€€', '€€€', '€€€€']
        
        try:
            return prices[prices.index(min_price):prices.index(max_price) + 1]
        except ValueError:
            print("Invalid price range provided.")
            return []
    
    def search(self, restaurantName=None, city=None, cuisineType=None,
               minPrice='€', maxPrice='€€€€', region=None,
               creditCard=None, facilitiesServices=None):
        """
        Searches for restaurants based on various optional filters.
        
        Args:
            restaurantName (str): Name of the restaurant.
            city (str): City where the restaurant is located.
            cuisineType (str): Type of cuisine the restaurant serves.
            minPrice (str): Minimum price range symbol (e.g., '€').
            maxPrice (str): Maximum price range symbol (e.g., '€€€€').
            region (str): Region where the restaurant is located.
            creditCard (str): Credit card acceptance filter.
            facilitiesServices (str): Facilities and services filter.
            
        Returns:
            DataFrame with filtered restaurant data.
        """
        df = self.df.copy()
        resultname = set([name for name in df['restaurantName']])
        
        # Apply filters for various criteria
        if restaurantName:
            resultname.intersection_update(self.get_restaurants('restaurantName', restaurantName))
        if city:
            resultname.intersection_update(self.get_restaurants('city', city))
        if cuisineType:
            resultname.intersection_update(self.get_restaurants('cuisineType', cuisineType))
        if region:
            resultname.intersection_update(self.get_restaurants('region', region))
        if creditCard:
            resultname.intersection_update(self.get_restaurants('creditCard', creditCard))
        if facilitiesServices:
            resultname.intersection_update(self.get_restaurants('facilitiesServices', facilitiesServices))
        
        # Apply price range filter
        price_range = self.get_price_range(minPrice, maxPrice)
        if price_range:
            resultname.intersection_update(self.get_restaurants('priceRange', price_range))
        
        # Filter the dataframe based on the results
        df = df[df['restaurantName'].isin(resultname)]
        return df[['restaurantName', 'address', 'cuisineType', 'priceRange', 'website']]
    
    def get_restaurants(self, column, query):
        """
        Retrieves a list of restaurant names based on a search query for a specific column.
        
        Args:
            column (str): Column name to search in (e.g., 'city', 'cuisineType').
            query (str): Query string to search for in the specified column.
            
        Returns:
            List of restaurant names that match the search criteria.
        """
        stemmer = PorterStemmer()
        if isinstance(query, str):
            query = [stemmer.stem(query.lower())]
        else:
            # If query is a list, stem each term
            query = [stemmer.stem(word.lower()) for word in query]  # Stem the query for more flexible matching

        json_path = f'dataset/{column}_index.json'
        restaurants = []
        
        with open(json_path) as f:
            try:
                json_file = json.load(f)
                for q in query:
                    restaurants += json_file[q]
                f.close()
            except:
                return set()
        return set(restaurants)

