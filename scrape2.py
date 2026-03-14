import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch data using the arXiv API
def fetch_arxiv_api_data(query, num_results=300):
    url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={num_results}'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to retrieve data from API")
        return []

    # Parse the XML response from the API
    root = ET.fromstring(response.content)
    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    
    abstracts = []
    for entry in entries:
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        abstracts.append((title, summary))

    print(f"Found {len(abstracts)} abstracts from API for {query}")
    return abstracts

# Function to scrape abstracts from arXiv using BeautifulSoup
def scrape_arxiv_abstracts(region, num_abstracts=300):
    url = f"https://arxiv.org/search/?query={region}&searchtype=all&abstracts=show&order=-announced_date_first&size={num_abstracts}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    print(f"Requesting: {url}")
    
    if response.status_code != 200:
        print(f"Failed to retrieve data, status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    abstracts = []
    
    for entry in soup.find_all('li', class_='arxiv-result'):
        title = entry.find('p', class_='title').get_text(strip=True)
        abstract = entry.find('span', class_='abstract').get_text(strip=True)
        abstracts.append((title, abstract))

    print(f"Found {len(abstracts)} abstracts from scraping for {region}")
    return abstracts

# Function to scrape abstracts for each region, first using the API, then fall back to scraping
def scrape_abstracts_for_regions(regions, num_abstracts=300):
    data = []
    
    for region in regions:
        print(f"Attempting to fetch abstracts for {region} using API...")
        abstracts = fetch_arxiv_api_data(region, num_abstracts)
        
        # If no data is found via API, fall back to scraping
        if not abstracts:
            print(f"API failed for {region}, attempting scraping...")
            abstracts = scrape_arxiv_abstracts(region, num_abstracts)
        
        # Append region label to each entry
        for title, abstract in abstracts:
            data.append([region, title, abstract])
    
    # Save data to CSV
    df = pd.DataFrame(data, columns=['Region', 'Title', 'Abstract'])
    df.to_csv('research_abstracts.csv', index=False)
    print("Data saved to 'research_abstracts.csv'.")

# List of regions to scrape
regions = ['astronomy', 'psychology', 'sociology']

# Scrape and save abstracts
scrape_abstracts_for_regions(regions)
