import os
from bs4 import BeautifulSoup
import json
import csv
import numpy as np
class Parser:
    """
    Class for parsing restaurant information from HTML files and saving it to a TSV file.
    """

    def __init__(self):
        pass

    def parse_restaurant(self, html_file_path):
        """
        Parses a single HTML file to extract restaurant information.

        Args:
            html_file_path (str): Path to the HTML file.

        Returns:
            dict: Dictionary containing the extracted restaurant information.
        """
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the JSON-LD script tag
        json_ld_script = soup.find('script', type='application/ld+json')

        if json_ld_script:
            # Parse the JSON-LD
            json_ld_data = json.loads(json_ld_script.string)

            # Find and parse relevant HTML elements
            text_blocks = soup.find_all(class_='data-sheet__block--text')
            price_and_type = self.parse_price_and_typeCuisine(text_blocks)
            services_section = soup.find('div', class_='restaurant-details__services')
            services_facilities = self.parse_facilities(services_section)
            website_element = soup.find(class_='collapse__block-item link-item')
            website = self.parse_website(website_element) if website_element else None
            description = soup.find('div', class_='data-sheet__description').get_text(strip=True)

            # Extract important information with default values of None
            restaurant_info = {
                "restaurantName": json_ld_data.get("name", None),
                "address": json_ld_data.get("address", {}).get("streetAddress", None),
                "city": json_ld_data.get("address", {}).get("addressLocality", None),
                "postalCode": json_ld_data.get("address", {}).get("postalCode", None),
                "country": json_ld_data.get("address", {}).get("addressCountry", None),
                "region": json_ld_data.get("address", {}).get("addressRegion", None),
                "priceRange": price_and_type[0] if price_and_type else None,
                "cuisineType": price_and_type[1] if price_and_type else None,
                "description": description if description else None,
                "facilitiesServices": services_facilities if services_facilities else None,
                "creditCards": self.parse_credit_cards(json_ld_data.get("paymentAccepted", "")),
                "phoneNumber": json_ld_data.get("telephone", None),
                "website": website,
                "latitude": json_ld_data.get("latitude", None).astype(np.float64),
                "logitude": json_ld_data.get("longitude", None).astype(np.float64),
            }

            return restaurant_info
        else:
            print("No JSON-LD found.")
            return None

    def show_restaurant_info(self, info):
        """
        Prints the extracted restaurant information.

        Args:
            info (dict): Dictionary containing restaurant information.
        """
        for key, value in info.items():
            print(f"{key}: {value}")

    def parse_credit_cards(self, cards):
        """
        Parses credit card information from a string.

        Args:
            cards (str): Comma-separated string of credit card names.

        Returns:
            list: List of simplified credit card names.
        """
        possible_cards = {
            "American Express": 'Amex',
            "Mastercard": 'Mastercard',
            "Visa": 'Visa'
        }
        results = []

        card_list = [card.strip() for card in cards.split(',')]

        for card in card_list:
            for key in possible_cards.keys():
                if key in card:
                    results.append(possible_cards[key])
                    break  # Stop checking once a match is found
        return results

    def parse_price_and_typeCuisine(self, blocks):
        """
        Parses price and cuisine type from text blocks.

        Args:
            blocks (list): List of BeautifulSoup elements containing text.

        Returns:
            list: List containing price and cuisine type.
        """
        results = []

        # Iterate over the blocks and filter for the desired content
        for block in blocks:
            text_content = block.get_text(strip=True)
            if '€' in text_content or 'Cuisine' in text_content:
                results.append(text_content)

        result = list(set(results))[0]
        result = result.split("·")
        result[0] = result[0].replace(" ", "").replace("\n", "")
        result[1] = result[1].replace("\n", "").lstrip()
        return result

    def parse_facilities(self, services_section):
        """
        Parses facilities and services from a section.

        Args:
            services_section (BeautifulSoup element): Element containing services information.

        Returns:
            list: List of facilities and services.
        """
        facilities = services_section.find_all('li')
        facilities_list = [facility.get_text(strip=True) for facility in facilities]
        return facilities_list

    def parse_website(self, block):
        """
        Parses the website URL from a block element.

        Args:
            block (BeautifulSoup element): Element containing the website link.

        Returns:
            str: Website URL.
        """
        a_tag = block.find('a')
        website = a_tag.get("href")
        return website

    def save_all_restaurant_info_to_tsv(self, root_folder, output_tsv):
        """
        Traverses a directory to parse all HTML files and save the extracted
        information to a TSV file.

        Args:
            root_folder (str): Root folder containing HTML files.
            output_tsv (str): Output TSV file path.
        """
        all_restaurant_info = []

        # Traverse the directory
        for folder_name, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith('.html'):
                    file_path = os.path.join(folder_name, filename)
                    restaurant_info = self.parse_restaurant(file_path)
                    if restaurant_info:
                        all_restaurant_info.append(restaurant_info)

        # Write all restaurant info to a TSV file
        if all_restaurant_info:
            with open(output_tsv, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=all_restaurant_info[0].keys(), delimiter='\t')
                writer.writeheader()
                for info in all_restaurant_info:
                    writer.writerow({key: ', '.join(value) if isinstance(value, list) else value for key, value in info.items()})

        print(f"Data saved to {output_tsv}")