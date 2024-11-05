import requests
import os
import threading

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