from jupyter_ai_agent_magics import jupyter_utils
from jupyter_ai_agent_magics.tools import jupyter_tools
from jupyter_ai_agent_magics.tools import human_tools

def get_tools():
    server_url, token = jupyter_utils.get_server_url_and_token()
    jupyter_agent_tools = jupyter_tools.JupyterNotebookTools(server_url, token)
    human_agent_tools = human_tools.HumanTools()
    return jupyter_agent_tools.get_tools() + human_agent_tools.get_tools()