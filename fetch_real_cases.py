"""
Fetch real case URLs from CanLII for the scraper
"""
import requests
from bs4 import BeautifulSoup
import csv
from pathlib import Path
import time
import random

OUTPUT_CSV = Path("real_cases_to_scrape.csv")

def fetch_recent_cases():
    print("="*60)
    print("FETCHING REAL CANLII CASE URLS")
    print("="*60)
    
    # CanLII Ontario Superior Court recent decisions
    # We will try a few pages or months if possible, but CanLII structure is complex.
    # We'll try the "nav/en/on/onsc/recent/schedule.html" or similar?
    # Actually, let's try a search query URL which returns a list.
    # Search for "wrongful dismissal" in 2024
    
    base_search_url = "https://www.canlii.org/en/on/onsc/"
    # CanLII doesn't allow easy scraping of search results typically (dynamic).
    # But we can try the monthly lists.
    
    # https://www.canlii.org/en/on/onsc/nav/date/2024/
    
    urls_found = []
    
    # Try getting 2024 cases from a few months
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for month in months:
        if len(urls_found) >= 100:
            break
            
        url = f"https://www.canlii.org/en/on/onsc/nav/date/2024/{month}/"
        print(f"Scanning {url}...")
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"  Failed: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find case links
            # Usually in a table with class="decision-table" or list
            # Links look like /en/on/onsc/doc/2024/2024onsc123/2024onsc123.html
            
            links = soup.find_all('a', href=True)
            page_links = []
            
            for link in links:
                href = link['href']
                if "/en/on/onsc/doc/2024/" in href:
                    full_url = "https://www.canlii.org" + href
                    if full_url not in urls_found and full_url not in page_links:
                        page_links.append(full_url)
            
            print(f"  Found {len(page_links)} cases")
            urls_found.extend(page_links)
            
            time.sleep(1 + random.random())
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Prepare CSV
    print(f"\nFound {len(urls_found)} total unique URLs")
    
    rows = []
    for i, url in enumerate(urls_found):
        # Generate a CaseID
        case_id = url.split("/")[-1].replace(".html", "")
        rows.append({
            "Caseid": case_id,
            "URL": url,
            "Outcome": "Unknown" # Placeholder
        })
    
    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Caseid", "URL", "Outcome"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    fetch_recent_cases()
