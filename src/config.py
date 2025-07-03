import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Set OpenAI API key
def set_openai_api_key():
    """
    Load the OpenAI API key from environment variables and set it for the OpenAI library.

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment variables.')
    openai.api_key = api_key 

# Get LLM model name from environment or use default
def get_llm_model_name():
    """
    Returns the LLM model name from the environment variable LLM_MODEL_NAME, or 'gpt-4o-mini' if not set.
    """
    return os.getenv('LLM_MODEL_NAME', 'gpt-4o-mini') 