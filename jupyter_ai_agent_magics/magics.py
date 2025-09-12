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
        self._server_cache_dict = {}

    def get_server(self, use_tools=True):
        cache_key = (
            config.ConfigManager.instance.model_provider, 
            config.ConfigManager.instance.model_name, 
            use_tools)
        if cache_key not in self._server_cache_dict:
            logger.info(
                "creating llm server %s with model %s use_tools %s", 
                config.ConfigManager.instance.model_provider,
                config.ConfigManager.instance.model_name,
                use_tools
            )
            server = LLMServer(
                model_provider=config.ConfigManager.instance.model_provider,
                model_name=config.ConfigManager.instance.model_name,
                api_key=config.ConfigManager.instance.api_key,
                project_id=config.ConfigManager.instance.project_id,
                location=config.ConfigManager.instance.location,
                tools=agent_tools.get_tools() if use_tools else []
            )
            self._server_cache_dict[cache_key] = server
            return server
        return self._server_cache_dict[cache_key]
        
    @cell_magic
    #@magic_arguments()
    #@argument('--context', '-c', help='Additional context for the agent')
    def agent(self, line, cell):
        """Execute code with AI agent assistance"""
#        args = parse_argstring(self.ai_agent, line)
#        if line:
#            print("line: ", line)
#        print("cell:", cell)
        
        self.get_server().process_message_html(cell)

    @cell_magic
    def chat(self, line, cell):
        """Run LLM without access to any tools."""
        self.get_server(use_tools=False).process_message_html(cell)



def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(AgentMagics) 