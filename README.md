# News Scraper

## Project Description
A Python project to scrape news articles from URLs, summarize them, identify topics using GenAI (OpenAI GPT via Langchain), and enable semantic search over the results using a vector database (Chroma).

## Requirements
- Python 3.9 or higher

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

## Dependencies
- [Langchain](https://python.langchain.com/)
- [OpenAI](https://platform.openai.com/docs/api-reference)
- [Chroma](https://docs.trychroma.com/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- [Requests](https://docs.python-requests.org/)
- [pytest](https://docs.pytest.org/) (for testing)
- [tqdm](https://tqdm.github.io/) (progress bars)
- [colorama](https://pypi.org/project/colorama/) (colored output)

## Environment Variables

The project requires an OpenAI API key. You can provide it in two ways:

1. **Using a `.env` file** (recommended):
   - Create a `.env` file in your project root with the following content:
     ```
     OPENAI_API_KEY=your-openai-api-key-here
     ```
   - Make sure to install `python-dotenv` if you want automatic loading, or manually load the `.env` in your shell before running scripts.

2. **Or set the environment variable manually:**
   - On Windows:
     ```
     set OPENAI_API_KEY=your-openai-api-key-here
     ```
   - On Unix/Mac:
     ```
     export OPENAI_API_KEY=your-openai-api-key-here
     ```

## Features
- Scrape news articles from URLs
- Summarize articles and extract topics using GenAI (OpenAI GPT)
- Store and search articles semantically using a vector database (Chroma)
- **Progress bars** for scraping, processing, and searching (tqdm)
- **Colored output** for status and errors (colorama)
- **Improved error messages** and user feedback
- **CLI options** for output file, force re-scraping, top-k results, and similarity threshold
- **Topics are cleaned** to remove numbers/bullets and stored as a list in the article JSON

## How to Run

### News Scraper Usage

From the project root, use the following command to scrape and summarize articles:

```
python -m src.news_scraper "<url or urls.txt>" [--output OUTPUT] [--force]
```

**Examples:**
```
python -m src.news_scraper https://example.com/news1
python -m src.news_scraper urls.txt --output my_articles.json
python -m src.news_scraper urls.txt --force
```
- `--output`: Specify the output JSON file (default: articles.json)
- `--force`: Force re-scraping even if URLs exist in the output file

### Semantic Search Usage

After scraping and summarizing articles, perform semantic search using:

```
python -m src.semantic_search "your search query" [--top_k N] [--threshold T] [--output OUTPUT]
```

**Examples:**
```
python -m src.semantic_search "climate change"
python -m src.semantic_search "tennis" --top_k 10 --threshold 0.8
python -m src.semantic_search "Wimbledon" --output search_results.json
```
- `--top_k`: Number of results to return (default: 1)
- `--threshold`: Similarity threshold (0-1, default: 0.3). Only results above this score are shown.
- `--output`: Save search results to a JSON file

**Output:**
- Results are printed in a readable block format (not a table) for clarity, especially for long summaries/topics.
- The script shows how many results were found above the threshold vs. requested.

## Input File Example

If using a file for URLs, create a `urls.txt` with one URL per line:
```
https://example.com/news1
https://example.com/news2
```

## Output

Scraped and summarized articles are saved in `articles.json` (or your chosen output file) in the project root. 
- Each article includes: `url`, `headline`, `content`, `summary`, and `topics` (as a list).
- Topics are cleaned to remove numbers/bullets for better semantic search.

## Running Tests

To run automated tests:
```
pytest
```

## Notes on Topics and Semantic Search
- Topics are stored as a list in the article JSON, but for semantic search, all topics are still embedded as a single string along with the summary.
- This means searching for a single topic may be less effective if the article has many topics. For best results, consider storing each topic as a separate document in the vector DB.

## Troubleshooting
- **Missing API Key:** Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.
- **Network Errors:** Check your internet connection and that the URLs are accessible.
- **Import Errors:** Make sure you are running commands from the project root and the virtual environment is activated.
- **Chroma DB Issues:** If you encounter issues with the vector database, try deleting the `chroma_db/` directory and rerunning the scripts.

## License
This project is for educational and demonstration purposes. 