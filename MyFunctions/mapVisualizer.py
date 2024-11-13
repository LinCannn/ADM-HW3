import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

class RestaurantMapVisualizer:
    def __init__(self, enhanced_searcher, original_file, mapbox_token):
        """
        Initializes the visualizer with an enhanced search engine, original file, and Mapbox token.
        
        Args:
            enhanced_searcher: Instance of EnhancedSearchEngine
            original_file: Path to the original file with restaurant data
            mapbox_token: Mapbox API token for accessing high-resolution maps
        """
        self.enhanced_searcher = enhanced_searcher
        self.df = pd.read_csv(original_file, sep='\t')
        self.mapbox_token = mapbox_token

    def plot_restaurants(self, top_k=20, price_color_mapping=None, query="", cuisine=None, price_range=None, facilities=None):
        """
        Visualizes the top-k restaurants on a Mapbox map.
        
        Args:
            top_k: Number of restaurants to display
            price_color_mapping: Dictionary mapping price ranges to colors
            query: Search query (optional)
            cuisine: Type of cuisine (optional)
            price_range: Price range (optional)
            facilities: List of facilities (optional)
        """
        if price_color_mapping is None:
            price_color_mapping = {
                '€€€€': 'rgb(255, 0, 0)',
                '€€€': 'rgb(255, 165, 0)',
                '€€': 'rgb(255, 255, 0)',
                '€': 'rgb(0, 255, 0)'
            }

        try:
            # Perform enhanced search
            search_results = self.enhanced_searcher.enhanced_search(
                query=query, k=top_k, cuisine=cuisine, price_range=price_range, facilities=facilities
            )

            # Merge with coordinates from the original DataFrame
            results_with_coords = pd.merge(
                search_results,
                self.df[['restaurantName', 'latitude', 'logitude']],
                on='restaurantName',
                how='left'
            )

            # Set up the Mapbox map
            fig = go.Figure(go.Scattermapbox())

            # Set up layout with dynamic centering based on average latitude/longitude
            fig.update_layout(
                mapbox=dict(
                    accesstoken=self.mapbox_token,
                    style="mapbox://styles/mapbox/streets-v11",
                    center={
                        "lat": results_with_coords['latitude'].mean(),
                        "lon": results_with_coords['logitude'].mean()
                    },
                    zoom=5
                ),
                width=900,
                height=700,
            )

            # Add restaurant markers
            for _, restaurant in results_with_coords.iterrows():
                if pd.isna(restaurant['latitude']) or pd.isna(restaurant['logitude']):
                    continue
                
                lat = restaurant['latitude']
                lon = restaurant['logitude']
                price = restaurant.get('priceRange', '€')
                restaurant_name = restaurant['restaurantName']
            
                # Tooltip information
                tooltip_text = f"{restaurant_name}\nPrice: {price}"
                if 'cuisineType' in restaurant:
                    tooltip_text += f"\nCuisine: {restaurant['cuisineType']}"
                if 'similarity_score' in restaurant:
                    tooltip_text += f"\nScore: {restaurant['similarity_score']:.2f}"
                
                color = price_color_mapping.get(price, 'gray')
                fig.add_trace(go.Scattermapbox(
                    lon=[lon],
                    lat=[lat],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    text=tooltip_text,
                    hoverinfo="text",
                    name=price,
                    showlegend=False
                ))

            # Add legend markers explicitly for price ranges
            for price, color in price_color_mapping.items():
                fig.add_trace(go.Scattermapbox(
                    lon=[None], 
                    lat=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=price,
                    showlegend=True 
                ))

            fig.show()

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise

