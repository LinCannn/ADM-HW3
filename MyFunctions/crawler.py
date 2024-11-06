import requests
import os
import threading
import random

class Crawler:

    def __init__(self):
        #This trick came to me in a dream by Linus Torvalds
        self.user_agents = [
            # Safari on Mac
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",

            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",

            # Firefox on Mac
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0; rv:91.0) Gecko/20100101 Firefox/91.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:92.0) Gecko/20100101 Firefox/92.0",

            # Edge on Windows
            "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2046.40",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.55",

            # Chrome on Linux
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",

            # iPhone Safari
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",

        ]

        self.headers = {
            'User-Agent': random.choice(self.user_agents)
        }


    def save_url_as_html(self, url, path):
        """
        Downloads the content of a given URL and saves it as an HTML file.
        
        Parameters:
            url (str): The URL of the web page to download.
            path (str): The directory where the HTML file will be saved.
        
        This function does not return anything but will print an error if the download fails.
        """
        try:
            response = requests.get(url,headers=self.headers)
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

    def parse_urls(self, file_path):
        """
        Reads a file containing URLs, groups them by page numbers, and returns a list of URL lists.
        
        Parameters:
            file_path (str): The path to the file containing the URLs and page numbers.
        
        Returns:
            list: A list where each element is a list of URLs belonging to a particular page.
        """
        page_urls = []  # List to hold lists of URLs for each page
        current_page_urls = []  # Temporary list to store URLs for the current page
        
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

    def save_all_as_html(self, fn):
        """
        Reads URLs from a file and saves each as an HTML file in a structured directory.
        
        Parameters:
            fn (str): The file path of the file containing URLs and page numbers.
        
        This function does not return anything but saves the HTML files in the 'restaurants_html' directory.
        """
        all_urls = self.parse_urls(fn)
        base_directory = 'restaurants_html'
        os.makedirs(base_directory, exist_ok=True)  # Create base directory if it doesn't exist

        for c, url_list in enumerate(all_urls, start=1):
            page_directory = os.path.join(base_directory, str(c))  # Corrected directory creation
            os.makedirs(page_directory, exist_ok=True)  # Create page directory if it doesn't exist

            # Use threads to save URLs concurrently
            threads = []
            for u in url_list:
                thread = threading.Thread(target=self.save_url_as_html, args=(u, page_directory))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
    
    def count_files(self, path):
        count = 0
        for root_dir, cur_dir, files in os.walk(path): #Let's check if we got all the files
            count += len(files)
        return count
