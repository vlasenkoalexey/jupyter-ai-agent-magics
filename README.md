# Jupyter AI Agent Magics  🪐+🤖+✨=❤️

Jupyter AI Agent Magics brings AI agent capabilities to JupyterLab, enhancing its role as the ultimate research tool for AI research, which has historically lacked robust LLM integration. This project enables seamless interaction with large language models within JupyterLab, streamlining workflows for researchers and data scientists. Drawing inspiration from projects like Jupyter AI, Jupyter MCP Server, and Jupyter AI Agents, it provides practical tools to elevate your notebook experience.

## Demo

![Demo](resources/demo.gif)

## Benefits of AI Agent Integration in Jupyter Notebooks

Integrating AI agents directly into Jupyter notebooks provides numerous advantages for data scientists, analysts, and researchers:

1. **Immediate Assistance**: Get code suggestions, debugging help, and explanations without leaving your workflow
2. **Streamlined Development**: Reduce context switching between different applications when seeking help
3. **Knowledge Augmentation**: Access expertise across libraries, algorithms, and best practices on demand
4. **Educational Support**: Learn as you work with explanations and guidance tailored to your code
5. **Productivity Enhancement**: Complete analyses faster with automated code generation and optimization

## Why Jupyter's Cell-Based Structure Is Ideal for AI Agent Interaction

Jupyter notebooks' architecture of small, discrete code cells creates the perfect environment for AI agent collaboration:

1. **Granular Context**: Each cell provides a focused, manageable unit of code for the AI to analyze
2. **Iterative Interaction**: The cell-by-cell execution model matches AI's turn-based conversation pattern
3. **Progressive Development**: Both humans and AI can build solutions incrementally, one logical step at a time
4. **Immediate Feedback**: Results appear directly below each cell, creating tight feedback loops
5. **Perfect Scope Size**: Cells typically contain just enough code to perform a specific task without overwhelming complexity


## Features

Jupyter AI Agent Magics offers:
- Simple to use and understand `%%agent` and `%%chat` magics
- Prebuilt set of tools to modify and run Jupyter cells
- Great usability by implementing LLM response streaming, and user callbacks
- Great extensibility to offer option to easilty introduce
new tools for Agents to use and even allow Agents to generate tools on their own
- Great flexibility by integrating with top model providers
- Great hackability, the code is easy to understand and modify


## Installation

Due to underlying dependency issue https://github.com/datalayer/jupyter-nbmodel-client/issues/46 
have to pin jupyter related libraries to older versions.
It is a good idea to do this in an isolated environment like conda or venv.
Assuming that you have conda installed:


To install in editable/development mode:

```bash
# Create new environment
conda create --name py312 python=3.12
conda activate py312

# Install package in dev mode
pip install -e .

# Start jupyter as usual:
jupyter lab
```

## Configuration

Configuration is pretty typical, settings are configured throughout Jupyter restarts.
Following model providers are supported:
- Google
- Anthropic
- Anthropic via Vertex
- OpenAI

For automatic configuration open `configure.ipynb` notebook.

### Google

Model list is available here: https://ai.google.dev/gemini-api/docs/models
API key is configured at https://aistudio.google.com/app/apikey
It can be provided as environment variable:

```sh
export GOOGLE_API_KEY='your-api-key-here'
```

or explicitly through config manager.

```python
from jupyter_ai_agent_magics import config
config_manager = config.ConfigManager.instance

config_manager.model_provider = config.ModelProvider.Google
config_manager.model_name = "gemini-2.5-pro"
config_manager.api_key = "..."
```

### Anthropic

Model list is available here: https://docs.anthropic.com/en/docs/about-claude/models/overview
API key is configured at https://docs.anthropic.com/en/docs/get-started
It can be provided as environment variable:

```sh
export ANTHROPIC_API_KEY='your-api-key-here'
```

or explicitly through config manager.

```python
from jupyter_ai_agent_magics import config
config_manager = config.ConfigManager.instance

config_manager.model_provider = config.ModelProvider.Anthropic
config_manager.model_name = "claude-sonnet-4-20250514"
config_manager.api_key = "..."
```

### OpenAI

Model list is available here: https://platform.openai.com/docs/pricing
API key is configured at https://platform.openai.com/api-keys
It can be provided as environment variable:

```sh
export OPENAI_API_KEY='your-api-key-here'
```

or explicitly through config manager.
Note that recent GPT-5 models might requre organization verification.

```python
from jupyter_ai_agent_magics import config
config_manager = config.ConfigManager.instance

config_manager.model_provider = config.ModelProvider.OpenAI
config_manager.model_name = "gpt-4"
config_manager.api_key = ""
```

### Anthropic via Vertex

Setup is described in https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude
You'll need working Google Cloud Project, where you explicitly enable specific model.

To connect your project to your VM/devbox run:

```sh
gcloud auth application-default login
```

Model list is available here: https://docs.anthropic.com/en/docs/about-claude/models/overview
You need to specify your project and the region where model is available.
See this notebook to for the list of regions: https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/anthropic_claude_3_intro.ipynb#scrollTo=Y8X70FTSbx7U

```python
from jupyter_ai_agent_magics import config
config_manager = config.ConfigManager.instance

config_manager.model_provider = config.ModelProvider.AnthropicVertex
config_manager.model_name = "claude-sonnet-4@20250514"
config_manager.project_id = "<your project>"
config_manager.location = "us-east5"
```

## Usage

Register magic as:

```python
from jupyter_ai_agent_magics import magics
magics.load_ipython_extension(get_ipython())
```

See `autoregister_agent_magic.ipynb` notebook if you'd like to pre-register magic to be auto loaded at Jupyter startup.

Call into LLM and allow it to execute some actions on your behalf using `%%agent` magic:

```
%%agent
Write a notebook demonstrating how to train MNIST model.
```

If you just want to ask LLM some question, and you don't want it to call into any tools, use `%%chat` magic:

```
%%chat
Tell my about asyncio in Python.
```

