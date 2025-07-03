import json
import os
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from src.config import set_openai_api_key
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
from src.utils import load_json_file, save_json_file, print_colored

colorama_init(autoreset=True)

ARTICLES_FILE = os.path.join(os.path.dirname(__file__), '..', 'articles.json')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')

"""
semantic_search.py

Performs semantic search over news articles using Langchain, OpenAI embeddings, and Chroma vector store.
"""

# Load articles
def load_articles():
    """
    Load articles from articles.json.

    Returns:
        list[dict]: List of article dictionaries.
    """
    with open(ARTICLES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_documents(articles):
    """
    Convert articles to Langchain Document objects for embedding and search.

    Args:
        articles (list[dict]): List of article dictionaries.

    Returns:
        list[Document]: List of Langchain Document objects with metadata.
    """
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
    """
    Build or load a persistent Chroma vector store from documents and embeddings.

    Args:
        docs (list[Document]): List of Langchain Document objects.
        embeddings: Embedding function (e.g., OpenAIEmbeddings).

    Returns:
        Chroma: Langchain Chroma vector store instance, or None on error.
    """
    try:
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            # Get URLs already in the vectorstore
            existing_urls = set()
            try:
                # Chroma's get() returns all docs with their metadata
                all_docs = vectorstore.get()
                metadatas = all_docs.get('metadatas', [])
                for meta in metadatas:
                    if meta and isinstance(meta, dict) and 'url' in meta:
                        existing_urls.add(meta['url'])
            except Exception as e:
                print(f"Warning: Could not retrieve existing vectorstore URLs: {e}")
            # Only add docs whose URL is not already present
            new_docs = [doc for doc in docs if doc.metadata.get('url') not in existing_urls]
            if new_docs:
                print(f"Adding {len(new_docs)} new articles to the vector store.")
                vectorstore.add_documents(new_docs)
            else:
                print("No new articles to add to the vector store.")
            return vectorstore
        else:
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
            return vectorstore
    except Exception as e:
        print(f"Error building/loading vector store: {e}")
        return None

def semantic_search(vectorstore, query, top_k=3):
    """
    Perform semantic search over the vector store.

    Args:
        vectorstore: Chroma vector store instance.
        query (str): Search query.
        top_k (int): Number of top results to return.

    Returns:
        list[Document]: List of matching Langchain Document objects.
    """
    try:
        return vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []

def main():
    """
    Main entry point for the semantic search CLI. Loads articles, builds/loads vector store, and performs search.
    """
    parser = argparse.ArgumentParser(description='Semantic search over news articles (Langchain version)')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--top_k', type=int, default=1, help='Number of results to return')
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold (0-1, default: 0.75)')
    parser.add_argument('--output', type=str, default=None, help='Optional output file to save search results as JSON')
    args = parser.parse_args()

    set_openai_api_key()
    print(Fore.CYAN + f"\nUser Query: {args.query}" + Style.RESET_ALL)
    try:
        articles = load_json_file(ARTICLES_FILE, default=[])
    except Exception as e:
        print_colored(f"Error loading articles: {e}", Fore.RED)
        return
    print(Fore.CYAN + f"Loaded {len(articles)} articles." + Style.RESET_ALL)
    print(Fore.CYAN + "\nBuilding documents for embedding/search..." + Style.RESET_ALL)
    docs = []
    for article in tqdm(articles, desc='Building docs', ncols=80):
        try:
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
                print(Fore.YELLOW + f"Skipping article (empty content): {meta.get('url', '')}" + Style.RESET_ALL)
                continue
            docs.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            print(Fore.RED + f"Error building document for article: {e}" + Style.RESET_ALL)
    if not docs:
        print(Fore.RED + "No valid articles to index." + Style.RESET_ALL)
        return
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        print_colored(f"Error initializing OpenAIEmbeddings: {e}", Fore.RED)
        return
    print(Fore.CYAN + "\nLoading or updating vector store..." + Style.RESET_ALL)
    try:
        vectorstore = build_or_load_vectorstore(docs, embeddings)
    except Exception as e:
        print_colored(f"Error building/loading vector store: {e}", Fore.RED)
        return
    if not vectorstore:
        print_colored("Vector store could not be created or loaded.", Fore.RED)
        return
    print(Fore.CYAN + f"\nPerforming semantic search (top {args.top_k}, threshold {args.threshold})..." + Style.RESET_ALL)
    results = []
    try:
        if hasattr(vectorstore, 'similarity_search_with_score'):
            results_with_scores = vectorstore.similarity_search_with_score(args.query, k=args.top_k)
            # Chroma returns higher score = more similar (score in [0,1])
            filtered = [(doc, score) for doc, score in results_with_scores if score >= args.threshold]
            results = filtered
        else:
            print(Fore.YELLOW + "Warning: similarity_search_with_score not available, falling back to similarity_search without threshold filtering." + Style.RESET_ALL)
            docs_only = vectorstore.similarity_search(args.query, k=args.top_k)
            results = [(doc, None) for doc in docs_only]
    except Exception as e:
        print_colored(f"Error during semantic search: {e}", Fore.RED)
        return
    if not results:
        print_colored("No results found above the threshold. Try a different query or lower the threshold.", Fore.YELLOW)
        return
    print(Fore.CYAN + f"\nTop Results (showing {len(results)} of {args.top_k} requested above threshold {args.threshold}):" + Style.RESET_ALL)
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        print(Fore.YELLOW + f"\nResult {i+1}:" + Style.RESET_ALL)
        print(f"Headline: {meta.get('headline', '')}")
        print(f"URL: {meta.get('url', '')}")
        print(f"Summary: {meta.get('summary', '')}")
        print(f"Topics: {meta.get('topics', '')}")
        if score is not None:
            print(f"Score: {score:.3f}")
        print('-' * 60)
    if args.output:
        try:
            save_json_file(args.output, [doc.metadata for doc, _ in results])
        except Exception as e:
            print_colored(f"Error saving results to {args.output}: {e}", Fore.RED)

if __name__ == '__main__':
    main() 