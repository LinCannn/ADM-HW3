from nltk import word_tokenize

def searching(query, df, vocabulary, inverted_index):
    # Process the query terms
    query_text = word_tokenize(query.lower())  # Tokenize and lowercase query
    term_ids = []  # List to store term IDs for query text
    
    # Check each query term in the vocabulary
    for term in query_text:
        if term in vocabulary:  # Check if term is included in vocabulary
            term_ids.append(vocabulary[term])
    
    # Find restaurants containing all query terms
    if not term_ids:
        return []  # No terms matched in vocabulary 
    
    # Get the initial list of restaurant IDs that contain the first term
    ideal_restaurants = set(inverted_index.get(str(term_ids[0]), []))
    
    # Narrow down results for each additional term
    for term_id in term_ids[1:]:
        ideal_restaurants.intersection_update(inverted_index.get(str(term_id), []))
        
    
    # Retrieve restaurant details from the DataFrame
    result = []
    for restaurant_id in ideal_restaurants:
        restaurant = df.loc[df["restaurantName"] == restaurant_id].iloc[0]
        result.append({
            "restaurantName": restaurant["restaurantName"],
            "address": restaurant["address"],
            "description_filtered": restaurant["description_filtered"],
            "website": restaurant["website"]
        })
    
    return result