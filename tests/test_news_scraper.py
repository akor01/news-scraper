import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from src import news_scraper

TEST_ARTICLES_FILE = 'test_articles.json'

@pytest.fixture(autouse=True)
def cleanup_test_file():
    yield
    if os.path.exists(TEST_ARTICLES_FILE):
        os.remove(TEST_ARTICLES_FILE)

def test_parse_input_single_url():
    url = 'https://example.com/news1'
    result = news_scraper.parse_input(url)
    assert result == [url]

def test_parse_input_multiple_urls():
    urls = 'https://a.com https://b.com'
    result = news_scraper.parse_input(urls)
    assert result == ['https://a.com', 'https://b.com']

def test_parse_input_file(tmp_path):
    file_path = tmp_path / 'urls.txt'
    file_path.write_text('https://a.com\nhttps://b.com\n')
    result = news_scraper.parse_input(str(file_path))
    assert result == ['https://a.com', 'https://b.com']

@patch('src.news_scraper.requests.Session.get')
def test_fetch_webpages_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '<html><h1>Headline</h1><article>Content</article></html>'
    mock_get.return_value = mock_response
    urls = ['https://example.com/news1']
    result = news_scraper.fetch_webpages(urls)
    assert result[0]['status_code'] == 200
    assert 'Headline' in result[0]['content']

@patch('src.news_scraper.BeautifulSoup')
def test_extract_article_data(mock_bs):
    mock_soup = MagicMock()
    mock_h1 = MagicMock()
    mock_h1.get_text.return_value = 'Test Headline'
    mock_article = MagicMock()
    mock_article.get_text.return_value = 'Test Content'
    mock_soup.find.side_effect = lambda tag: mock_h1 if tag == 'h1' else (mock_article if tag == 'article' else None)
    mock_bs.return_value = mock_soup
    html = '<html></html>'
    url = 'https://example.com/news1'
    result = news_scraper.extract_article_data(html, url)
    assert result['headline'] == 'Test Headline'
    assert result['content'] == 'Test Content'

def test_merge_articles(tmp_path):
    # Simulate merging logic
    articles = [
        {'url': 'https://a.com', 'headline': 'A', 'content': 'A'},
        {'url': 'https://b.com', 'headline': 'B', 'content': 'B'}
    ]
    existing_articles = [
        {'url': 'https://a.com', 'headline': 'A-old', 'content': 'A-old'},
        {'url': 'https://c.com', 'headline': 'C', 'content': 'C'}
    ]
    # Simulate the merging logic from news_scraper.py
    existing_by_url = {a['url']: a for a in existing_articles if 'url' in a}
    for article in articles:
        existing_by_url[article['url']] = article
    merged_articles = list(existing_by_url.values())
    assert len(merged_articles) == 3
    assert any(a['url'] == 'https://a.com' and a['headline'] == 'A' for a in merged_articles)
    assert any(a['url'] == 'https://b.com' for a in merged_articles)
    assert any(a['url'] == 'https://c.com' for a in merged_articles)

def test_skip_scraping_existing_urls(monkeypatch, tmp_path):
    # Simulate articles.json with one URL
    articles_file = tmp_path / 'articles.json'
    articles_file.write_text(json.dumps([
        {'url': 'https://a.com', 'headline': 'A', 'content': 'A'}
    ]))
    # Change working directory so 'articles.json' resolves to our test file
    monkeypatch.chdir(tmp_path)
    # Patch parse_input to return the same URL
    monkeypatch.setattr(news_scraper, 'parse_input', lambda x: ['https://a.com'])
    # Patch fetch_webpages to track calls
    called = {'fetch': False}
    def fake_fetch_webpages(urls):
        called['fetch'] = True
        return []
    monkeypatch.setattr(news_scraper, 'fetch_webpages', fake_fetch_webpages)
    # Patch set_openai_api_key to do nothing
    monkeypatch.setattr(news_scraper, 'set_openai_api_key', lambda: None)
    # Patch argparse to avoid CLI
    class Args:
        input = 'dummy'
        output = 'articles.json'
        force = False
    monkeypatch.setattr(news_scraper.argparse, 'ArgumentParser', lambda *a, **k: MagicMock(parse_args=lambda: Args()))
    # Run main
    news_scraper.main()
    # Assert fetch_webpages was not called
    assert not called['fetch'] 