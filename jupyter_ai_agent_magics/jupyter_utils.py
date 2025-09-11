import os
import logging
import psutil
from jupyter_core.paths import jupyter_runtime_dir
import glob
from jupyter_server.utils import url_path_join
import requests
import json
from IPython.core.getipython import get_ipython
from jupyter_ai_agent_magics import ipynbname

logger = logging.getLogger(__name__)

def _get_parent_process_id():
    # Get current process ID
    current_pid = os.getpid()
    parent_pid = psutil.Process(current_pid).parent().pid
    return parent_pid

def _get_jupyter_env_info():
    server_info_file = glob.glob(jupyter_runtime_dir() + "/*" + str(_get_parent_process_id()) + "*.json")
    if not server_info_file:
        raise ValueError("Can't detect jupyter information")
    if len(server_info_file) != 1:
        logger.warning("more than one jupyter configs are found: %s", server_info_file)
    with open(server_info_file[0]) as f:
        return json.load(f)

def get_curent_notebook_info():
    kernel = get_ipython().kernel
    connection_file=kernel.config['IPKernelApp']['connection_file']
    with open(connection_file, 'r') as f:
        connection_info = json.load(f)
    return connection_info

def get_server_url_and_token():
    env_info = _get_jupyter_env_info()
    return env_info['url'], env_info['token']

def get_current_notebook_path() -> str:
    return str(ipynbname.path())

def get_current_notebook_name() -> str:
    return ipynbname.name()
