import argparse
import os
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin


def parse_input(input_value):
    # If input_value is a file and exists, read URLs from file
    if os.path.isfile(input_value):
        with open(input_value, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    # If input_value contains whitespace, treat as multiple URLs
    elif any(c.isspace() for c in input_value):
        urls = [u.strip() for u in input_value.split() if u.strip()]
        return urls
    # Otherwise, treat as a single URL
    else:
        return [input_value.strip()]


def fetch_webpages(urls):
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1"
    }
    session = requests.Session()
    for url in urls:
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"\nFetched: {url}\nStatus: {response.status_code}")
            print(response.text[:200] + ("..." if len(response.text) > 200 else ""))
            results.append({
                'url': url,
                'status_code': response.status_code,
                'content': response.text
            })
        except Exception as e:
            print(f"\nFailed to fetch {url}: {e}")
            results.append({
                'url': url,
                'status_code': None,
                'content': None,
                'error': str(e)
            })
    return results


def extract_article_data(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    # Headline: try <h1>
    headline = None
    h1 = soup.find('h1')
    if h1:
        headline = h1.get_text(strip=True)
    # Main content: try <article>, else largest <div> with text
    content = None
    article = soup.find('article')
    if article:
        content = article.get_text(separator=' ', strip=True)
    else:
        # Find the largest <div> by text length
        divs = soup.find_all('div')
        if divs:
            largest_div = max(divs, key=lambda d: len(d.get_text(strip=True)))
            content = largest_div.get_text(separator=' ', strip=True)
    return {
        'url': url,
        'headline': headline,
        'content': content
    }


def main():
    parser = argparse.ArgumentParser(description='News Scraper Input Handler')
    parser.add_argument('input', type=str, help='A single URL, a string of URLs, or a filename containing URLs (one per line)')
    args = parser.parse_args()

    urls = parse_input(args.input)
    print('Parsed URLs:')
    for url in urls:
        print(url)
    fetched = fetch_webpages(urls)
    articles = []
    for page in fetched:
        if page.get('content'):
            article_data = extract_article_data(page['content'], page['url'])
            articles.append(article_data)
    print('\nExtracted Articles:')
    print(json.dumps(articles, indent=2, ensure_ascii=False))
    # Save to JSON file
    output_file = 'articles.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f'\nSaved {len(articles)} articles to {output_file}')

if __name__ == '__main__':
    main() 