# News Scrapper

## Project Description
A Python project to scrape news articles from URLs, summarize them, identify topics using GenAI, and enable semantic search over the results.

## Setup
1. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/Mac:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- Place your Python scripts in the `src/` directory.
- Run your main script as needed.

## Features
- Scrape news articles from URLs
- Summarize articles and extract topics using GenAI
- Store and search articles semantically using a vector database

## How to Run

From the project root, use the following command:

```
python -m src.news_scraper "<url or urls.txt>"
```

This ensures the src package is recognized and all imports work correctly.

## Semantic Search Usage

After scraping and summarizing articles, you can perform semantic search using the following command from the project root:

```
python -m src.semantic_search "your search query"
```

Replace `"your search query"` with your desired search phrase. The script will return the most relevant articles based on semantic similarity.

## Environment Variables

Create a `.env` file in your project root with your OpenAI API key:
```
OPENAI_API_KEY=your-openai-api-key-here
```

## Running Tests

To run automated tests:
```
pytest
```

## Input File Example

If using a file for URLs, create a `urls.txt` with one URL per line:
```
https://example.com/news1
https://example.com/news2
```

## Output

Scraped and summarized articles are saved in `articles.json` in the project root. 