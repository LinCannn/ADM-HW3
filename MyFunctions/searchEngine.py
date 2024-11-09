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
            json_file = json.load(f)
            for q in query:
                restaurants += json_file[q]
            f.close()
        return set(restaurants)
