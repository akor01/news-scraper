import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch, MagicMock
from src import semantic_search

@pytest.fixture
def sample_articles():
    return [
        {'url': 'https://a.com', 'headline': 'A', 'summary': 'Apple news', 'topics': ['fruit', 'food'], 'content': 'Apple is a fruit.'},
        {'url': 'https://b.com', 'headline': 'B', 'summary': 'Banana news', 'topics': ['fruit', 'yellow'], 'content': 'Banana is yellow.'},
    ]

def test_build_documents(sample_articles):
    docs = semantic_search.build_documents(sample_articles)
    assert len(docs) == 2
    assert docs[0].metadata['headline'] == 'A'
    assert 'Apple news' in docs[0].page_content

@patch('src.semantic_search.Chroma')
@patch('src.semantic_search.OpenAIEmbeddings')
def test_build_or_load_vectorstore(mock_embeddings, mock_chroma, sample_articles):
    docs = semantic_search.build_documents(sample_articles)
    mock_chroma.return_value = MagicMock()
    mock_embeddings.return_value = MagicMock()
    vs = semantic_search.build_or_load_vectorstore(docs, mock_embeddings())
    assert vs is not None

@patch('src.semantic_search.Chroma')
@patch('src.semantic_search.OpenAIEmbeddings')
def test_semantic_search(mock_embeddings, mock_chroma, sample_articles):
    docs = semantic_search.build_documents(sample_articles)
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [MagicMock(metadata={'headline': 'A', 'url': 'https://a.com', 'summary': 'Apple news', 'topics': 'fruit, food'})]
    result = semantic_search.semantic_search(mock_vs, 'apple', 1)
    assert len(result) == 1
    assert result[0].metadata['headline'] == 'A' 