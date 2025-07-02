import argparse
import os
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from src.config import set_openai_api_key
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def parse_input(input_value):
    """
    Parse input to extract URLs.

    Args:
        input_value (str): A single URL, a string of URLs, or a filename containing URLs (one per line).

    Returns:
        list[str]: List of URLs.
    """
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
    """
    Fetch the HTML content of each URL.

    Args:
        urls (list[str]): List of URLs to fetch.

    Returns:
        list[dict]: List of dicts with keys 'url', 'status_code', 'content', and optional 'error'.
    """
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
    """
    Extract the headline and main content from the HTML of a news article.

    Args:
        html (str): HTML content of the page.
        url (str): URL of the article.

    Returns:
        dict: Dictionary with 'url', 'headline', and 'content'.
    """
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


def summarize_and_identify_topics(article):
    """
    Use Langchain's ChatOpenAI to generate a summary and identify main topics for the article.

    Args:
        article (dict): Article dictionary with at least a 'content' field.

    Returns:
        tuple: (summary (str or None), topics (list[str] or None))
    """
    content = article.get('content')
    if not content:
        return None, None
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        # Summarization
        summary_prompt = ChatPromptTemplate.from_template(
            """
            Summarize the following news article in 2-4 sentences, focusing on the key points:
            {article_text}
            """
        )
        summary_chain = summary_prompt | llm
        summary_resp = summary_chain.invoke({"article_text": content})
        summary = summary_resp.content.strip() if isinstance(summary_resp.content, str) else str(summary_resp.content)
        # Topic identification
        topics_prompt = ChatPromptTemplate.from_template(
            """
            List 3-5 main topics or keywords that best describe the following news article:
            {article_text}
            """
        )
        topics_chain = topics_prompt | llm
        topics_resp = topics_chain.invoke({"article_text": content})
        topics_raw = topics_resp.content.strip() if isinstance(topics_resp.content, str) else str(topics_resp.content)
        # Parse topics (split by line or comma)
        if '\n' in topics_raw:
            topics = [t.strip('- ').strip() for t in topics_raw.split('\n') if t.strip()]
        else:
            topics = [t.strip() for t in topics_raw.split(',') if t.strip()]
        return summary, topics
    except Exception as e:
        print(f"Langchain LLM error: {e}")
        return None, None


def main():
    """
    Main entry point for the news scraper CLI. Parses input, fetches articles, summarizes and identifies topics, and saves/merges results.
    """
    parser = argparse.ArgumentParser(description='News Scraper Input Handler')
    parser.add_argument('input', type=str, help='A single URL, a string of URLs, or a filename containing URLs (one per line)')
    args = parser.parse_args()

    set_openai_api_key()
    urls = parse_input(args.input)
    print('Parsed URLs:')
    for url in urls:
        print(url)
    fetched = fetch_webpages(urls)
    articles = []
    for page in fetched:
        if page.get('content'):
            article_data = extract_article_data(page['content'], page['url'])
            summary, topics = summarize_and_identify_topics(article_data)
            article_data['summary'] = summary
            article_data['topics'] = topics
            articles.append(article_data)
    print('\nExtracted Articles:')
    print(json.dumps(articles, indent=2, ensure_ascii=False))
    # Save to JSON file
    output_file = 'articles.json'

    # Load existing articles if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_articles = json.load(f)
            except Exception:
                existing_articles = []
    else:
        existing_articles = []

    # Create a dict for fast lookup by URL
    existing_by_url = {a['url']: a for a in existing_articles if 'url' in a}
    for article in articles:
        existing_by_url[article['url']] = article  # update or add

    # Save merged articles
    merged_articles = list(existing_by_url.values())
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_articles, f, ensure_ascii=False, indent=2)
    print(f'\nSaved {len(merged_articles)} articles to {output_file}')

if __name__ == '__main__':
    main() 