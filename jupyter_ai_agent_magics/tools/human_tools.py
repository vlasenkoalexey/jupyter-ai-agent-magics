from typing import List, Dict, Any, Callable, Optional, Union
import os
import re
import logging
import inspect
import nbformat
from uuid import uuid4

from langgraph.types import Command, interrupt
from langchain_core.tools import StructuredTool

from jupyter_ai_agent_magics import jupyter_utils

# Corrected logger initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from threading import Event

event = Event()

class HumanTools:
    def __init__(self):
        pass

    def ask_human(self, query: str) -> str:
        """Ask human a clarifying question.

        Args:
            query (str): The question or task to present to the human.

        Returns:
            str: The human's response.
        """
        return "stop execution, response will be provided in next query"

    def ask_human_yes_or_no(self, query: str) -> str:
        """Ask human a question that should have either yes or no answer.

        Args:
            query (str): The question or task to present to the human.

        Returns:
            str: The human's response, either yes or no.
        """
        return "stop execution, response will be provided in next query"

    def ask_human_to_choose_option(self, query: str, options:list[str]) -> str:
        """Ask human to choose one of the options in the proposed list.

        Args:
            query (str): The question or task to present to the human.
            options (str): List of options for human to choose.

        Returns:
            str: The human's response, should be one of the proposed options.
        """
        return "stop execution, response will be provided in next query"    

    def get_tools(self) -> List[StructuredTool]:
        """Return a list of tools that this class provides.

        Returns:
            List[StructuredTool]: List of structured tools
        """
        return [
            StructuredTool.from_function(
                func=self.ask_human,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.ask_human_yes_or_no,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.ask_human_to_choose_option,
                parse_docstring=True
            ),            
        ]
