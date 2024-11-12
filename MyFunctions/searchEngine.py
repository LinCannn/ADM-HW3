from nltk import word_tokenize
from nltk.stem import PorterStemmer
import json
import pandas as pd
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SearchEngine:
    def __init__(self, original, df_path, vocabulary_path, inverted_index_path):
        self.original_df = pd.read_table(original)
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
        result = self.original_df[self.original_df['restaurantName'].isin(ideal_restaurants)]
        return result[['restaurantName', 'address', 'description', 'website']]
        
    def tf(self, df, vocabulary):
        word_freq_dict = {}
        for idx, row in df.iterrows():
            description = row["description_filtered"].split(" ")
            restaurant_name = row["restaurantName"]
            word_counts = pd.Series(description).value_counts()
            word_counts = word_counts[word_counts.index.isin(vocabulary)]
            
            for word, count in word_counts.items():
                word_freq_dict.setdefault(word, []).append((restaurant_name, count / len(description)))
        
        return pd.DataFrame(list(word_freq_dict.items()), columns=["word", "restaurant_frequencies"])
    
    def idf(self, tf, total_documents):
        tf['idf'] = tf['restaurant_frequencies'].apply(lambda x: math.log(total_documents / (1 + len(x))))
        return tf
    
    def tf_idf_score(self, df1):
        df1 = self.df[self.df['restaurantName'].isin(df1['restaurantName'])]
        vocabulary = set(df1['description_filtered'].str.cat(sep=' ').split(" "))
        term_frequency = self.tf(df1, vocabulary)
        inverse_df = self.idf(term_frequency, df1.shape[0])
        inverse_df['tf-idf'] = inverse_df['restaurant_frequencies'].apply(
            lambda row: [(restaurant, freq * row['idf']) for restaurant, freq in row])
        return inverse_df[['word', 'tf-idf']]
    
    def compute_query_tf_idf(self, query):
        # Tokenize and stem query text
        query_tokens = word_tokenize(query.lower())
        stemmer = PorterStemmer()
        query_tokens = [stemmer.stem(word) for word in query_tokens]
        
        # Compute TF for the query
        term_freq = {}
        for term in query_tokens:
            if term in self.vocabulary:
                term_freq[term] = term_freq.get(term, 0) + 1
        
        # Normalize term frequency (TF)
        query_length = len(query_tokens)
        query_tf = {term: freq / query_length for term, freq in term_freq.items()}
        
        # Compute IDF for each term in query
        query_idf = {}
        for term in query_tf:
            if term in self.vocabulary:
                term_id = self.vocabulary[term]
                # IDF calculation for query terms
                doc_count = len(self.inverted_index[str(term_id)])
                query_idf[term] = math.log(len(self.df) / (1 + doc_count))
        
        # Compute TF-IDF for the query
        query_tf_idf = {term: query_tf[term] * query_idf[term] for term in query_tf}
        return query_tf_idf

    def compute_cosine_similarity(self, query_tf_idf, restaurant_tf_idf):
        # Prepare the query and restaurant TF-IDF vectors
        query_vector = np.array([query_tf_idf.get(word, 0) for word in restaurant_tf_idf.keys()])
        restaurant_vector = np.array(list(restaurant_tf_idf.values()))
        
        # Calculate cosine similarity
        return cosine_similarity([query_vector], [restaurant_vector])[0][0]
    
    def get_restaurant_scores(self, query, k = 5):
        # Compute TF-IDF for the query
        query_tf_idf = self.compute_query_tf_idf(query)
        
        # Compute the TF-IDF for all restaurant descriptions
        restaurant_scores = []
        for idx, row in self.df.iterrows():
            restaurant_name = row['restaurantName']
            description = row['description_filtered']
            description_tokens = description.split(' ')
            restaurant_tf_idf = {}
            
            # Compute TF-IDF for the restaurant description
            term_freq = {}
            for term in description_tokens:
                if term in self.vocabulary:
                    term_freq[term] = term_freq.get(term, 0) + 1
            restaurant_tf = {term: freq / len(description_tokens) for term, freq in term_freq.items()}
            
            # Compute IDF for the restaurant description
            for term in restaurant_tf:
                if term in self.vocabulary:
                    term_id = self.vocabulary[term]
                    doc_count = len(self.inverted_index[str(term_id)])
                    restaurant_tf_idf[term] = restaurant_tf[term] * math.log(len(self.df) / (1 + doc_count))
            
            # Compute cosine similarity between the query and restaurant
            similarity = self.compute_cosine_similarity(query_tf_idf, restaurant_tf_idf)
            restaurant_scores.append((restaurant_name, similarity))
        
        # Sort restaurants by similarity
        restaurant_scores = sorted(restaurant_scores, key=lambda x: x[1], reverse=True) #List of tuples (name, score)

        #########CHANGE HERE######

        #get table and only get top 5

        ###################à
        return restaurant_scores


class AdvancedSearchEngine:
    def __init__(self):
        # Load the original dataset
        self.original_df = pd.read_table("dataset/restaurant_info.tsv")
    
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
        df = self.original_df.copy()
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

