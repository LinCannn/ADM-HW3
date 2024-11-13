import heapq
from typing import List, Dict
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
import pandas as pd
from MyFunctions.searchEngine import SearchEngine


class EnhancedSearchEngine:
    def __init__(self, original_file: str, vocabulary_file: str, inverted_index_file: str):
        """
        Initialize the enhanced search engine with all necessary files and data structures.
        """
        # Load the original search engine components
        self.base_searcher = SearchEngine(original_file, vocabulary_file, inverted_index_file)
        
        # Load the complete dataset for additional features
        self.df = pd.read_csv(original_file, sep='\t')
        
        # Weight parameters for scoring, all equal
        self.weights = {
            'description': 0.25,
            'cuisine': 0.25,
            'facilities': 0.25,
            'price': 0.25
        }

    # Evaluate the score for the restaurant based on our criteria 
    def calculate_custom_score(self, 
                             row_index: int,
                             query: str, #?
                             cosine_scores: dict,  # Pass the scores as a parameter
                             desired_cuisine: str = None,
                             desired_price: str = None,
                             desired_facilities: List[str] = None) -> float:
        """
        Calculate a custom score for a restaurant based on multiple criteria.
        """
        restaurant = self.df.loc[row_index]
        final_score = 0.0

        # 1. Get the description score from the pre-calculated cosine scores
        restaurant_name = restaurant['restaurantName']
        description_score = cosine_scores.get(restaurant_name, 0.0)
        final_score = description_score * self.weights['description']
        
        # 2. Cuisine match
        if desired_cuisine and hasattr(restaurant, 'cuisineType'):
            if desired_cuisine.lower() == str(restaurant['cuisineType']).lower():
                cuisine_score = 1.0 * self.weights['cuisine']
                final_score += cuisine_score
        
        # 3. Facilities match
        if desired_facilities and hasattr(restaurant, 'facilitiesServices'):
            facilities = str(restaurant['facilitiesServices']).lower().split(',')
            facilities = [f.strip() for f in facilities]
            matches = sum(1 for f in desired_facilities if f.lower() in facilities)
            if len(desired_facilities) > 0:
                facilities_score = (matches / len(desired_facilities)) * self.weights['facilities']
                final_score += facilities_score


        # 4. Price range match
        if desired_price and hasattr(restaurant, 'priceRange'):
            restaurant_price = len(str(restaurant['priceRange']).strip())
            desired_price_level = len(desired_price)
            if restaurant_price == desired_price_level:
                price_score = 1.0 * self.weights['price']
                final_score += price_score
            
        return final_score

    # Define the enanched_search function
    def enhanced_search(self, 
                       query: str, 
                       k: int = 10, 
                       cuisine: str = None, 
                       price_range: str = None, 
                       facilities: List[str] = None) -> pd.DataFrame:
        """
        Perform an enhanced search with custom scoring and return top-k results.
        """
        # Get base search results
        base_results = self.base_searcher.search(query)
        
        # Get cosine similarity scores once and convert to dictionary for faster lookup
        cosine_scores_list = self.base_searcher.get_restaurant_scores(query)
        cosine_scores = {name: score for name, score in cosine_scores_list}
        
        # Create a list to store scores
        scores = []
        
        # Calculate custom scores for each result
        for idx in base_results.index:
            custom_score = self.calculate_custom_score(
                idx,
                query,
                cosine_scores,  # Pass the pre-calculated scores
                cuisine,
                price_range,
                facilities
            )
            scores.append((custom_score, idx))
        
        # Sort by score and get top k
        scores.sort(reverse=True)
        top_k_scores = scores[:k]
        
        # Get the indices of top k results
        top_indices = [idx for score, idx in top_k_scores]
        
        # Get the full data for these restaurants
        results = base_results.loc[top_indices].copy()
        
        # Add the custom scores
        results['similarity_score'] = [score for score, _ in top_k_scores]
        
        # Add additional information from the original dataset
        if 'cuisineType' in self.df.columns:
            results['cuisineType'] = self.df.loc[top_indices, 'cuisineType'].values
        if 'priceRange' in self.df.columns:
            results['priceRange'] = self.df.loc[top_indices, 'priceRange'].values
        if 'facilitiesServices' in self.df.columns:
            results['facilitiesServices'] = self.df.loc[top_indices, 'facilitiesServices'].values
        
        return results

    # Define print_result function to print our results
    def print_results(self, results: pd.DataFrame):
        """
        Print search results in a formatted way.
        """
        for i, (idx, row) in enumerate(results.iterrows(), 1):
            print(f"\n{i}. {row['restaurantName']} (Score: {row['similarity_score']:.4f})")
            print(f"Address: {row['address']}")
            print(f"Website: {row['website']}")
            print(f"Description: {row['description'][:200]}...")
            print("-" * 80)



class RestaurantSearchInterface:
    def __init__(self, enhanced_searcher, original_file: str ):
        """
        Initialize the search interface with the enhanced search engine
        """
        self.enhanced_searcher = enhanced_searcher
        self.df = pd.read_csv(original_file, sep='\t')
        
        # Define lists of options

        # Cusine_types
        self.cuisine_types = list(self.df['cuisineType'].dropna().unique().tolist())
        self.cuisine_types.sort()
        self.cuisine_types.insert(0,'All Cuisines')

        # Facilities
        self.facilities = self.df['facilitiesServices'].dropna().str.split(',').explode().str.strip().unique().tolist()

        # Price_ranges
        self.price_ranges = [
            'All Prices',
            '€',
            '€€',
            '€€€',
            '€€€€'
        ]
        
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Search text input
        self.search_text = widgets.Text(
            description='Search:',
            placeholder='Enter keywords (e.g., greek, pasta...)',
            layout=widgets.Layout(width='50%')
        )
        
        # Result select slider
        self.results_select = widgets.IntSlider(
            min=1,
            max=20,
            step=1,
            value=5,
            description='Show top:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )

        # Cuisine type dropdown
        self.cuisine_select = widgets.Dropdown(
            options=self.cuisine_types,
            description='Cuisine:',
            value='All Cuisines',
            layout=widgets.Layout(width='50%')
        )
        
        # Price range dropdown
        self.price_select = widgets.Dropdown(
            options=self.price_ranges,
            description='Price:',
            value='All Prices',
            layout=widgets.Layout(width='50%')
        )
        
        # Facilities multiple selection using Checkboxes
        self.facilities_boxes = [
            widgets.Checkbox(
                value=False,
                description=facility,
                layout=widgets.Layout(width='auto')
            ) for facility in self.facilities
        ]
        
        # Create a FlexBox layout for facilities
        self.facilities_box = widgets.VBox([
            widgets.HTML("<h3>Select Facilities:</h3>"),
            widgets.Box(self.facilities_boxes, 
                       layout=widgets.Layout(
                           display='flex',
                           flex_flow='row wrap',
                           width='100%'
                       ))
        ])
        
        # Search button
        self.search_button = widgets.Button(
            description='Search',
            button_style='primary',
            layout=widgets.Layout(width='20%')
        )
        
        # Clear button
        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            layout=widgets.Layout(width='20%')
        )
        
        # Output area for search results
        self.output = widgets.Output()
        
        # Bind button clicks to handlers
        self.search_button.on_click(self.handle_search)
        self.clear_button.on_click(self.handle_clear)

    def format_results_as_table(self, results):
        """Format search results as an HTML table"""
        table_style = """
        <style>
        .search-results {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        .search-results th, .search-results td {
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .search-results th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .search-results tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .search-results tr:hover {
            background-color: #f0f0f0;
        }
        </style>
        """
        
        # Start the table
        html = table_style + '<table class="search-results">'
        
        # Add headers
        headers = ['Name', 'Score',  'Address', 'Website', 'Description']
        html += '<tr>' + ''.join([f'<th>{h}</th>' for h in headers]) + '</tr>'
        
        # Add rows
        for _, row in results.iterrows():
            html += '<tr>'
            # Name
            html += f'<td>{row["restaurantName"]}</td>'
            # Score
            html += f'<td>{row["similarity_score"]:.4f}</td>'
            # Address
            html += f'<td>{row["address"]}</td>'
            # Website
            html += f'<td>{row["website"]}</td>'
            # Description (truncated)
            desc = row['description'][:200] + '...' if len(row['description']) > 200 else row['description']
            html += f'<td>{desc}</td>'
            html += '</tr>'
            
        
        html += '</table>'
        return html
        
    def handle_search(self, button):
        """Handle search button click"""
        with self.output:
            clear_output()
            
            # Get search parameters
            query = self.search_text.value.strip()
            if not query:
                print("Please enter a search query.")
                return
                
            cuisine = None if self.cuisine_select.value == 'All Cuisines' else self.cuisine_select.value
            price_range = None if self.price_select.value == 'All Prices' else self.price_select.value
            
            # Get selected facilities from checkboxes
            facilities = [f.description for f in self.facilities_boxes if f.value]
            facilities = facilities if facilities else None
            
            k = self.results_select.value
            
            try:
                # Perform search
                results = self.enhanced_searcher.enhanced_search(
                    query=query,
                    k=k,
                    cuisine=cuisine,
                    price_range=price_range,
                    facilities=facilities
                )
                
                # Display results
                if len(results) == 0:
                    print("No results found.")
                else:
                    display(HTML(self.format_results_as_table(results)))
                    
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    
    def handle_clear(self, button):
        """Handle clear button click"""
        self.search_text.value = ''
        self.cuisine_select.value = 'All Cuisines'
        self.price_select.value = 'All Prices'
        # Clear all facility checkboxes
        for checkbox in self.facilities_boxes:
            checkbox.value = False
        self.results_select.value = 5
        with self.output:
            clear_output()
    
    def display_interface(self):
        """Display the complete search interface"""
        # Create layout
        search_box = widgets.VBox([
            widgets.HTML("<h2>Restaurant Search</h2>"),
            widgets.HBox([self.search_text, self.results_select]),
            self.cuisine_select,
            self.price_select,
            self.facilities_box,
            widgets.HBox([self.search_button, self.clear_button]),
            self.output
        ])
        
        # Display the search interface
        display(search_box)
