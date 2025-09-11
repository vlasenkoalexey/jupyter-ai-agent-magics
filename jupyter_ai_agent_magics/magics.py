import os
import json
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import magic_arguments, parse_argstring
from IPython.utils.capture import capture_output
from IPython.display import display, HTML
import logging

from jupyter_ai_agent_magics.llm_server import LLMServer

from jupyter_ai_agent_magics.tools import agent_tools
from functools import partial

from jupyter_ai_agent_magics import config

logger = logging.getLogger('jupyter_ai_agent_magics')

@magics_class
class AgentMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        print("registering magic, creating llm server:", config.ConfigManager.instance.model_provider)
        self.server = LLMServer(
            model_provider=config.ConfigManager.instance.model_provider,
            model_name=config.ConfigManager.instance.model_name,
            api_key=config.ConfigManager.instance.api_key,
            project_id=config.ConfigManager.instance.project_id,
            location=config.ConfigManager.instance.location,
            tools=agent_tools.get_tools()
        )        

    @cell_magic
    #@magic_arguments()
    #@argument('--context', '-c', help='Additional context for the agent')
    def agent(self, line, cell):
        """Execute code with AI agent assistance"""
#        args = parse_argstring(self.ai_agent, line)
        if line:
            print("line: ", line)
        print("cell:", cell)
        self.server.process_message_html(cell)

def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(AgentMagics) 