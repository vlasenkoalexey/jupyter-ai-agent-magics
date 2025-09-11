from typing import Dict, List, Optional, Any, Literal, Generator


def create_model(
    model_provider: Literal["vertex_anthropic", "anthropic", "openai", "google"] = "vertex_anthropic",
    model_name: str = "claude-3-opus-20240229",
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = "us-central1",
    temperature: float = None,
):
    if model_provider == "vertex_anthropic":
        if not project_id:
            raise ValueError("project_id is required for AnthropicVertex client")
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        model = ChatAnthropicVertex(
            model=model_name, project=project_id, location=location, temperature=temperature
        )
    elif model_provider == "openai":
        # pip install langchain-openai
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it in the configuration.")
        from langchain_openai import ChatOpenAI  # Import OpenAI LLM
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
    elif model_provider == "anthropic":
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable or provide it in the configuration.")
        from langchain_community.chat_models import ChatAnthropic
        model = ChatAnthropic(
            anthropic_api_key=self.api_key, model=model_name, temperature=temperature
        )
    elif model_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not temperature:
            temperature = 0.7
        
        # pip install -U langchain-google-genai
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or provide it in the configuration.")
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=2,
            api_key=api_key
        )
    else:
        raise ValueError(f"model_provider {model_provider} is not supported")
        
    return model