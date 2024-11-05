import requests
import os
import threading
from bs4 import BeautifulSoup
import json

def save_url_as_html(url, path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Create a filename based on the URL
        filename = url.split('/')[-1] or 'index'
        if not filename.endswith('.html'):
            filename += '.html'

        # Save the content to an HTML file
        with open(os.path.join(path, filename), 'w', encoding='utf-8') as html_file:
            html_file.write(response.text)

    except Exception as e:
        print(f"Failed to save {url}: {e}")

def parse_urls(file_path):
    page_urls = []
    current_page_urls = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():  # Check if the line contains a page number
                if current_page_urls:  # If current_page_urls is not empty, add it to page_urls
                    page_urls.append(current_page_urls)
                    current_page_urls = []  # Reset for the next page
            elif line:  # If the line is a URL
                current_page_urls.append(line)

        # Add the last collected page URLs if any
        if current_page_urls:
            page_urls.append(current_page_urls)

    return page_urls

def save_all_as_html(fn):
    all_urls = parse_urls(fn)
    base_directory = 'restaurants_html'
    os.makedirs(base_directory, exist_ok=True)  # Create base directory if it doesn't exist

    for c, url_list in enumerate(all_urls, start=1):
        page_directory = os.path.join(base_directory, str(c))  # Corrected directory creation
        os.makedirs(page_directory, exist_ok=True)  # Create page directory if it doesn't exist

        # Use threads to save URLs concurrently
        threads = []
        for u in url_list:
            thread = threading.Thread(target=save_url_as_html, args=(u, page_directory))
            threads.append(thread)
            thread.start()
            print(f'N of threads working: {len(threads)}\n')

        # Wait for all threads to complete
        for thread in threads:
            thread.join()


def parse_restaurant(html_file_path):
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
        price_and_type = parse_price_and_typeCuisine(text_blocks)
        services_section = soup.find('div', class_='restaurant-details__services')
        services_facilities = parse_facilities(services_section)
        website_element = soup.find(class_='collapse__block-item link-item')
        website = parse_website(website_element) if website_element else None

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
            "description": json_ld_data.get("review", {}).get("description", None),
            "facilitiesServices": services_facilities if services_facilities else None,
            "creditCards": parse_credit_cards(json_ld_data.get("paymentAccepted", "")),
            "phoneNumber": json_ld_data.get("telephone", None),
            "website": website,
        }

        return restaurant_info
    else:
        print("No JSON-LD found.")
    

def show_restaurant_info(info):
        # Print the extracted information
        for key, value in info.items():
            print(f"{key}: {value}")



def parse_credit_cards(cards):
    possible_cards = {
        "American Express": 'Amex',
        "Mastercard": 'Mastercard',
        "Visa": 'Visa'
    }
    
    results = []

    card_list = [card.strip() for card in cards.split(',')]
    
    for card in card_list:
        for key in possible_cards.keys():
            # Check if the key (full card name) is in the card string
            if key in card:
                results.append(possible_cards[key])
                break  # Stop checking once a match is found
    return results

def parse_price_and_typeCuisine(blocks):
    results = []

    # Iterate over the blocks and filter for the desired content
    for block in blocks:
        text_content = block.get_text(strip=True)
    
        # Check if the text contains '€' or 'Cuisine' to filter desired entries
        if '€' in text_content or 'Cuisine' in text_content:
            results.append(text_content)

    result = list(set(results))[0]
    result = result.split("·")
    result[0] = result[0].replace(" ","").replace("\n","")
    result[1] = result[1].replace("\n","").lstrip()
    return result

def parse_facilities(services_section):
    facilities = services_section.find_all('li')
    facilities_list = [facility.get_text(strip=True) for facility in facilities]
    return facilities_list

def parse_website(block):
    a_tag = block.find('a')
    # Extract the href attribute
    website = a_tag.get("href")
    return website


def save_all_restaurant_info_to_tsv(root_folder, output_tsv):
    # List to store all restaurant info dictionaries
    all_restaurant_info = []

    # Traverse the directory
    for folder_name, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.html'):  # Check if the file is an HTML file
                file_path = os.path.join(folder_name, filename)
                restaurant_info = parse_restaurant(file_path)
                if restaurant_info:  # Add only if data was extracted
                    all_restaurant_info.append(restaurant_info)

    # Write all restaurant info to a TSV file
    if all_restaurant_info:
        with open(output_tsv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=all_restaurant_info[0].keys(), delimiter='\t')
            writer.writeheader()  # Write the header row
            for info in all_restaurant_info:
                writer.writerow({key: ', '.join(value) if isinstance(value, list) else value for key, value in info.items()})

    print(f"Data saved to {output_tsv}")