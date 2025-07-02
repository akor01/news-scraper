import json
import os
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from src.config import set_openai_api_key

ARTICLES_FILE = os.path.join(os.path.dirname(__file__), '..', 'articles.json')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')

# Load articles
def load_articles():
    with open(ARTICLES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_documents(articles):
    docs = []
    for article in articles:
        summary = article.get('summary', '')
        topics = ', '.join(article.get('topics', [])) if isinstance(article.get('topics'), list) else (article.get('topics') or '')
        content = summary + '\n' + topics
        meta = {
            'url': article.get('url'),
            'headline': article.get('headline'),
            'summary': summary,
            'topics': topics,
        }
        if not content.strip():
            print(f"Skipping article (empty content): {meta.get('url', '')}")
            continue
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def build_or_load_vectorstore(docs, embeddings):
    try:
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        else:
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
            return vectorstore
    except Exception as e:
        print(f"Error building/loading vector store: {e}")
        return None

def semantic_search(vectorstore, query, top_k=3):
    try:
        return vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Semantic search over news articles (Langchain version)')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--top_k', type=int, default=3, help='Number of results to return')
    args = parser.parse_args()

    set_openai_api_key()
    articles = load_articles()
    print(f"Loaded {len(articles)} articles.")
    docs = build_documents(articles)
    if not docs:
        print("No valid articles to index.")
        return
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        print(f"Error initializing OpenAIEmbeddings: {e}")
        return
    vectorstore = build_or_load_vectorstore(docs, embeddings)
    if not vectorstore:
        print("Vector store could not be created or loaded.")
        return
    results = semantic_search(vectorstore, args.query, args.top_k)
    if not results:
        print("No results found.")
        return
    print("\nTop results:")
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\nResult {i+1}:")
        print(f"Headline: {meta.get('headline', '')}")
        print(f"URL: {meta.get('url', '')}")
        print(f"Summary: {meta.get('summary', '')}")
        print(f"Topics: {meta.get('topics', '')}")

if __name__ == '__main__':
    main() 