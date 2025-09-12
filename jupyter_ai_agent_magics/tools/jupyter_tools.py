from typing import List, Dict, Any, Callable, Optional, Union
import os
import re
import logging
import inspect
import nbformat
from uuid import uuid4
from langchain_core.tools import StructuredTool
from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import NbModelClient, get_jupyter_notebook_websocket_url
import jupyter_nbmodel_client.model # Keep this specific import if needed elsewhere
from IPython import get_ipython
from IPython.utils.capture import capture_output

import asyncio
import tenacity
import inspect
from pathlib import Path
import os


# https://github.com/datalayer/jupyter-nbmodel-client/blob/main/jupyter_nbmodel_client/client.py
# https://github.com/datalayer/jupyter-nbmodel-client/blob/main/jupyter_nbmodel_client/model.py

from jupyter_ai_agent_magics import jupyter_utils

logger = logging.getLogger(__name__)
#print("jupyter utils logger name:", __name__)
#logger.setLevel(logging.DEBUG)

class JupyterNotebookTools:
    """
    A class providing tools for interacting with Jupyter notebooks programmatically.

    This class encapsulates functionality for creating, reading, and manipulating
    Jupyter notebooks, executing code cells, and managing notebook content.
    It provides integration with LangChain via the get_tools() method.

    Attributes:
        server_url (str): The URL of the Jupyter server.
        token (str): Authentication token for the Jupyter server.
        kernel: Instance of KernelClient connected to the server.
        changes_origin: Unique identifier for tracking changes in notebooks.
    """

    # TODO: figure out if we can connect to existing kernel
    def __init__(self, server_url="http://localhost:8880", token="my-token"):
        """
        Initialize the JupyterNotebookTools class with server URL and token.

        Args:
            server_url (str): The Jupyter server URL.
            token (str): Authentication token for the Jupyter server.
        """
        self._notebook_clients = dict()
        self.server_url = server_url
        self.token = token
        self.changes_origin = hash(uuid4().hex)

        # Initialize kernel client
        logging.info("initializing KernelClient for server: %s", self.server_url)
        self.kernel = KernelClient(server_url=self.server_url, token=self.token)
        if inspect.iscoroutinefunction(self.kernel.start):
            logging.info("staring KernelClient async")
            asyncio.run(self.kernel.start())
        else:
            logging.info("staring KernelClient")
            self.kernel.start()

    def __del__(self):
        if self.kernel and self.kernel.is_alive():
            if inspect.iscoroutinefunction(self.kernel.stop):
                logging.info("stopping KernelClient async")
                asyncio.run(self.kernel.stop())
            else:
                logging.info("stopping KernelClient")
                self.kernel.stop()

    def _notebook_cell_id_to_index(self, notebook: NbModelClient, cell_id: str) -> Optional[int]:
        """Finds the index of a cell given its ID within a NbModelClient instance."""
        # Note: NbModelClient acts like a list of cells
        for cell_index, cell in enumerate(notebook):
            # Access cell ID correctly for NbModelClient's representation
            if cell.get("id") == cell_id: # Use .get() for safety
                return cell_index
        return None
    
    def _normalize_path_to_jupyter_root(path: str) -> str:
        if os.path.sep in path:
            return path
        env_info = jupyter_utils._get_jupyter_env_info()
        jupyter_root_dir = env_info["root_dir"]
        current_dir = os.getcwd()

        dir_relative_to_root = Path(os.getcwd()).relative_to(env_info["root_dir"])
        
        return str(dir_relative_to_root / path)

    def _start_notebook_client(self, notebook_path: str) -> NbModelClient:
        """
        Creates, starts, and returns a retried NbModelClient instance.
        Ensures the notebook client is stopped on exit using a context manager.
        """
        logger.info(">>>>>>>> Attempting to start notebook client for: %s", notebook_path)
        logger.info(">>>>> current working directory: %s", os.getcwd())

        if not os.path.exists(notebook_path):
            logger.error("Notebook %s does not exist", notebook_path)
            raise FileNotFoundError(f"Notebook '{notebook_path}' does not exist")
        
        notebook_path = JupyterNotebookTools._normalize_path_to_jupyter_root(notebook_path)

        logger.info("Normalized path respecive to notebook root: %s", notebook_path)

        if notebook_path in self._notebook_clients:
            logging.info("Returning notebook client from cache for %s:", notebook_path)
            return self._notebook_clients[notebook_path]
        
        logging.info("Creating new notebook client for %s:", notebook_path)
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(
                server_url=self.server_url, token=self.token, path=notebook_path
            )
        )
        self._notebook_clients[notebook_path] = notebook
        
        try:
            notebook.start()
            logger.debug("Notebook client started successfully for: %s", notebook_path)
            return notebook
        except Exception:
             logger.error("Failed to start notebook client after retries for: %s", notebook_path, exc_info=True)
             # Ensure cleanup even if start fails within tenacity retries (though tenacity should handle exceptions)
             if notebook:
                 notebook.stop()
             raise # Re-raise the exception caught by tenacity

    def _stop_notebook_client(self, notebook: NbModelClient):
        logger.debug("Attempting to stop notebook client")
        if notebook:
            try:
                with capture_output():
                    notebook.stop()
                logger.debug("Notebook stopped successfully")
            except Exception:
                logger.error("Failed to stop notebook client after retries", exc_info=True)
                raise

    def create_notebook(self, path: str) -> str:
        """
        Create a new Jupyter notebook with Python 3 kernel.

        Creates a new .ipynb file with default Python 3 kernel specification.
        The notebook contains no cells initially.

        Args:
            path (str): The full name with path where the notebook will be created.

        Returns:
            str: Status confirming that notebook is created.

        Raises:
            ValueError: If the name is empty.
        """
        if not path:
            raise ValueError("Notebook path is required")

        if os.path.exists(path):
            return f"Notebook {path} already exists"

        base_path = os.path.dirname(path)
        if base_path and not os.path.exists(base_path):
            logging.info(
                "Creating base directory %s for notebook %s",
                base_path, path)
            os.makedirs(base_path)

        notebook_info = jupyter_utils.get_curent_notebook_info()

        kernel_name = notebook_info["kernel_name"]

        # Define the notebook structure with specified fields
        notebook_dict = {
            'cells': [{
                'cell_type': 'code',
                'execution_count': None,
                'id': str(uuid4()),  # Generate a unique ID for the cell
                'metadata': {},
                'outputs': [],
                'source': ''
            }],
            'metadata': {
                'kernelspec': {
                    'display_name': f'Python ({kernel_name})',
                    'language': 'python',
                    'name': kernel_name
                },
                'language_info': {
                    'codemirror_mode': {'name': 'ipython', 'version': 3},
                    'file_extension': '.py',
                    'mimetype': 'text/x-python',
                    'name': 'python',
                    'nbconvert_exporter': 'python',
                    'pygments_lexer': 'ipython3',
                    'version': '3.11.11'  # todo: figure out if this nees updating
                }
            },
            'nbformat': 4,
            'nbformat_minor': 5
        }

        # Convert dictionary to NotebookNode object
        notebook_node = nbformat.from_dict(notebook_dict) # Changed var name to avoid conflict

        # Write the notebook to file
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook_node, f)

        return f"Successfully created {path} notebook"

    def read_notebook(self, path: str) -> nbformat.NotebookNode: # Corrected return type hint
        """
            Reads a Jupyter notebook file from the specified path and returns its contents.

            This method opens and parses a .ipynb file using nbformat, returning the notebook
            object which contains all cells, metadata, and other notebook information.

            Args:
                path (str): The full file path to the Jupyter notebook (.ipynb) file.

            Returns:
                nbformat.NotebookNode: The parsed notebook object containing the notebook's
                    structure, cells, and metadata.

            Raises:
                FileNotFoundError: If the specified path does not exist.
                nbformat.reader.NotJSONError: If the file is not a valid JSON-formatted notebook.
                IOError: If there are issues reading the file (e.g., permissions).
                UnicodeDecodeError: If the file cannot be decoded with UTF-8 encoding.
        """
        with open(path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
            # Ensure we return the actual NotebookNode, not its string representation
            return notebook

    def get_project_details(self, path: str = "") -> Dict[str, Any]:
        """
        Get details of all files in a project directory.

        Scans a directory recursively and returns information about all files,
        distinguishing between notebook files and other file types.
        Hidden files and temporary files (.tmp, .bak) are excluded.

        This method can return a lot of information prefer using notebook related methods when applicable.

        Args:
            path (str, optional): The directory path to scan. Defaults to current directory.

        Returns:
            Dict[str, Any]: Project information containing:
                - status: "success" if scan completed
                - files: List of file details with name, path, and type
                - project_path: The path that was scanned
        """
        files = []
        for root, subdir, filenames in os.walk(path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                if not (
                    filename.startswith('.') or
                    filename.endswith('.tmp') or
                    filename.endswith('.bak') or
                    full_path.startswith('./.')):
                    files.append({
                        "path": full_path,
                        "type": "notebook" if filename.endswith('.ipynb') else "file"
                    })

        return {
            "status": "success",
            "files": files,
            "project_path": path
        }

    def get_notebooks_overview(self, path: Union[str, List[str]] = ".", header_level: int = 2) -> Dict[str, Any]:
        """
        Get an overview of notebooks and extract markdown headers of a specific depth.

        Args:
            path (Union[str, List[str]]): Directory path, a file path, or a list of file paths.
            header_level (int): Markdown header level to extract (e.g., 1 for "#", 2 for "##"). Default is 2.

        Returns:
            Dict[str, Any]: Overview containing status, notebooks, and the scanned path.
        """
        def parse_notebook(file_path: str) -> Union[Dict[str, Any], None]:
            if not file_path.endswith('.ipynb'):
                logger.debug("Skipping non-.ipynb file: %s", file_path)
                return None
            if ".ipynb_checkpoints" in file_path:
                 logger.debug("Skipping checkpoint file: %s", file_path)
                 return None
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                    # Original regex was fine, keeping it:
                    # header_pattern = re.compile(rf"^\s*{'#' * header_level}\s+(.*)")
                    headers = []
                    for cell in nb.cells:
                        if cell.cell_type == "markdown":
                            for line in cell.source.splitlines():
                                # Using original refined regex
                                match = re.match(r"^\s*(#{1,%d})\s+(.*)" % header_level, line)
                                if match:
                                    # Keep original formatting
                                    full_header = f"{match.group(1)} {match.group(2).strip()}"
                                    headers.append(full_header)
                    result = {
                        "path": file_path,
                    }
                    if headers:
                        result["headers"] = headers
                    return result
            except Exception as e:
                # Keep original warning format
                logger.warning("Error reading notebook %s: %s", file_path, str(e))
                return None

        if isinstance(path, str):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
            if os.path.isfile(path):
                files = [path]
            else:
                files = [os.path.join(root, fname)
                         for root, _, filenames in os.walk(path)
                         for fname in filenames if fname.endswith('.ipynb')]
        elif isinstance(path, list):
            files = [p for p in path if isinstance(p, str) and os.path.exists(p)]
        else:
            raise TypeError("Path must be a string or a list of strings")

        notebooks = []
        for file_path in files:
            notebook = parse_notebook(file_path)
            if notebook:
                notebooks.append(notebook)

        return {
            "status": "success",
            "notebooks": notebooks,
            "project_path": path
        }

    def extract_output(self, output: dict) -> str:
        """
        Extract human-readable text from a Jupyter cell output.

        Processes different output types (display_data, execute_result, stream, error)
        and returns the appropriate text representation.

        Args:
            output (dict): Jupyter notebook cell output dictionary.

        Returns:
            str: Extracted text from the output.
        """
        # Keep original logic exactly
        if output["output_type"] == "display_data":
            # Assuming text/plain is always present for these types based on original code
            return output["data"]["text/plain"]
        elif output["output_type"] == "execute_result":
            return output["data"]["text/plain"]
        elif output["output_type"] == "stream":
            return output["text"]
        elif output["output_type"] == "error":
            # Returning the traceback list as is, might need .join('\n') if a single string is desired
            # Keeping original behavior which likely returns the list of strings
            return output["traceback"]
        else:
            return ""

    def get_current_notebook_path(self) -> str:
        """Returns a path of the currently opened notebook."""
        return jupyter_utils.get_current_notebook_path()

    def get_current_notebook_name(self) -> str:
        """Returns a name of the currently opened notebook."""
        return jupyter_utils.get_current_notebook_name()

    def get_current_notebook_cell_id(self) -> str:
        """Returns Id of the current cell."""
        notebook = get_ipython().kernel.shell
        cell = notebook.get_parent()
        return cell['metadata']['cellId']

    def add_execute_code_cell(self, cell_content: str, notebook_path: str) -> str:
        """
        Add a new code cell to a notebook and execute it immediately.

        Creates a new code cell with the provided content at the end of the notebook,
        executes it using the connected kernel, and returns the execution output.

        Args:
            cell_content (str): Python code content for the cell.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Cell execution output or empty string if no output was produced.
        """
        logger.info(f"Adding and executing code cell in: %s", notebook_path)
        logging.info(f"Adding and executing code cell content: %s", cell_content)
        notebook = None # Initialize notebook to None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Original logic from here
            cell_index = notebook.add_code_cell(cell_content)
            logger.info(f"Executing cell at index {cell_index}")
            notebook.execute_cell(cell_index, self.kernel)

            # Access outputs using the NbModelClient's structure
            # Assuming notebook._doc._ycells provides the necessary structure based on original code
            ydoc = notebook._doc
            # Need to ensure ydoc._ycells exists and cell_index is valid
            if ydoc and hasattr(ydoc, '_ycells') and cell_index < len(ydoc._ycells):
                 cell_data = ydoc._ycells[cell_index]
                 # Check if 'outputs' key exists in the cell data dictionary
                 outputs = cell_data.get("outputs", []) # Use .get for safety
            else:
                 logger.warning(f"Could not access outputs for cell index {cell_index} in {notebook_path}")
                 outputs = []


            # Original output extraction logic
            if len(outputs) == 0:
                cell_output_str = "" # Changed variable name slightly
            else:
                # Process list of outputs; handle potential non-string results from extract_output
                output_parts = [str(self.extract_output(output)) for output in outputs]
                cell_output_str = "\n".join(output_parts) # Join parts for a single string representation


            # Original return logic was implicitly returning the list or "",
            # Adjusting to return string as per docstring. If list is needed, change return type hint.
            return cell_output_str

        finally:
            pass
            #self._stop_notebook_client(notebook)


    def add_code_cell(self, cell_content: str, notebook_path: str) -> str:
        """
        Add a new code cell to a notebook.

        Creates a new code cell with the provided content at the end of the notebook.

        Args:
            cell_content (str): Python code content for the cell.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Success message with index and id of the added cell. (Changed from just id)
        """
        logger.debug(f"Adding a code cell to: {notebook_path}")
        notebook = None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Original logic
            cell_index = notebook.add_code_cell(cell_content)
            # Access cell ID safely
            cell_id = notebook[cell_index].get('id', 'unknown_id') if cell_index < len(notebook) else 'unknown_id'

            # Return format based on original code's implicit return in similar methods
            return f"Code cell added with index {cell_index} and id {cell_id}"
        except Exception as e:
            logger.error("Error adding new code cell", exc_info=True)
            raise

        finally:
            pass
            #self._stop_notebook_client(notebook)


    def add_markdown_cell(self, cell_content: str, notebook_path: str) -> str:
        """
        Add a new markdown cell to a notebook.

        Creates a new markdown cell with the provided content at the end of the notebook.

        Args:
            cell_content (str): Markdown formatted text content.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Success message confirming the cell was added with index and id.
        """
        logger.info(f"Adding markdown cell to: {notebook_path}")
        notebook = None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Original logic
            cell_index = notebook.add_markdown_cell(cell_content)
            # Access cell ID safely
            cell_id = notebook[cell_index].get('id', 'unknown_id') if cell_index < len(notebook) else 'unknown_id'

            # Original return format
            return f"Markdown cell added with index {cell_index} and id {cell_id}"

        finally:
            pass
            #self._stop_notebook_client(notebook)


    def update_code_cell_by_cell_id(self, cell_content: str, cell_id: str, notebook_path: str) -> str:
        """
        Update an existing code cell addressed by cell id with new content.

        Replaces the content of a code cell identified by its ID with new content.
        Does not execute the cell after updating.

        Args:
            cell_content (str): New Python code content for the cell.
            cell_id (string): Cell id to update.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Success message confirming the cell was updated.

        Raises:
            ValueError: If cell with the given id is not found or index is invalid.
        """
        logger.info(f"Updating code cell id {cell_id} in: {notebook_path}")
        notebook = None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Original logic to find index
            cell_index = self._notebook_cell_id_to_index(notebook, cell_id)
            if cell_index is None: # Check if index was found
                # No need to stop notebook here, finally block handles it
                raise ValueError(f"Cell with id {cell_id} is not found in {notebook_path}")

            # Original logic for update
            # The original code seemed complex, trying to replicate intent:
            # It sets source, checks index validity again (redundant?), creates a new cell object,
            # creates a ycell, and replaces the existing ycell in a transaction.

            # 1. Set source (optional? The ycell replacement might handle this)
            # notebook.set_cell_source(cell_index, cell_content) # Keeping original sequence

            # 2. Check index validity (already implicitly checked by finding it)
            # if cell_index >= len(notebook): # This check seems logically covered by finding the index
            #     raise ValueError(f"Cell with index {cell_index} is not found") # Redundant check

            # 3. Create new cell object and ycell
            # Ensure correct model API usage based on imports
            cell = jupyter_nbmodel_client.model.current_api.new_code_cell(source=cell_content, id=cell_id)
            # If metadata/outputs need preserving, logic would be more complex. Assume fresh cell state.
            ycell = notebook._doc.create_ycell(cell)

            # 4. Replace in transaction
            with notebook._lock:
                with notebook._doc._ydoc.transaction(origin=self.changes_origin):
                    # Ensure index is still valid before assignment
                    if cell_index < len(notebook._doc.ycells):
                         notebook._doc.ycells[cell_index] = ycell
                    else:
                         # This case should ideally not happen if index was valid initially
                         raise IndexError(f"Cell index {cell_index} became invalid during update operation for {notebook_path}")


            # 5. Get the ID of the *potentially* new cell state at that index
            # Re-fetch the ID after update to be sure, using safe access
            updated_cell_id = notebook[cell_index].get('id', 'unknown_id') if cell_index < len(notebook) else 'unknown_id'

            # Original return format
            return f"Code cell with index {cell_index} is updated, new cell id {updated_cell_id}" # Use the actual ID after update
        except Exception:
            logger.error("Failed to update_code_cell_by_cell_id", exc_info=True)
            raise            

        finally:
            pass
            #self._stop_notebook_client(notebook)


    def update_markdown_cell_by_id(self, cell_content: str, cell_id: str, notebook_path: str) -> str:
        """
        Update an existing markdown cell addressed by cell id with new content.

        Replaces the content of a markdown cell with the specified cell id.

        Args:
            cell_content (str): New markdown content for the cell.
            cell_id (str): ID of the markdown cell to update.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Success message confirming the cell was updated.

        Raises:
            ValueError: If cell with the given id is not found or index is invalid.
        """
        logger.info(f"Updating markdown cell id {cell_id} in: {notebook_path}")
        notebook = None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Find the index for the cell with the matching ID - Original logic
            cell_index = self._notebook_cell_id_to_index(notebook, cell_id)
            if cell_index is None:
                # No need to stop notebook here, finally block handles it
                raise ValueError(f"Markdown cell with id {cell_id} not found in {notebook_path}.")

            # Update cell content and type - Original logic sequence
            # 1. Set source (optional, see update_code_cell comments)
            # notebook.set_cell_source(cell_index, cell_content) # Keeping original sequence

            # 2. Check index validity (Redundant)
            # if cell_index >= len(notebook):
            #     raise ValueError(f"Cell index {cell_index} is invalid")

            # 3. Create new cell object and ycell
            # Ensure correct model API usage
            cell = jupyter_nbmodel_client.model.current_api.new_markdown_cell(source=cell_content, id=cell_id) # Preserve ID? Assume new ID like in code cell update
            new_cell_id = str(uuid4())
            cell['id'] = new_cell_id
            ycell = notebook._doc.create_ycell(cell)

            # 4. Replace in transaction
            with notebook._lock:
                with notebook._doc._ydoc.transaction(origin=self.changes_origin):
                     if cell_index < len(notebook._doc.ycells):
                         notebook._doc.ycells[cell_index] = ycell
                     else:
                          raise IndexError(f"Cell index {cell_index} became invalid during update operation for {notebook_path}")


            # 5. Get the actual ID after update
            updated_cell_id = notebook[cell_index].get('id', 'unknown_id') if cell_index < len(notebook) else 'unknown_id'

            # Original return format
            return f"Markdown cell with index {cell_index} has been updated. New cell ID: {updated_cell_id}" # Use actual ID

        finally:
            pass
            #self._stop_notebook_client(notebook)


    def execute_cell(self, cell_id: str, notebook_path: str) -> str:
        """
        Execute an existing cell in a notebook by its ID.

        Executes the cell identified by the specified ID using the connected kernel
        and returns the execution output as a string.

        Args:
            cell_id (str): Cell id of the cell to execute.
            notebook_path (str): Path to the notebook file.

        Returns:
            str: Concatenated cell execution output or empty string if no output was produced.

        Raises:
            ValueError: If cell with the given id is not found.
        """
        logger.debug(f"Executing cell id {cell_id} in: {notebook_path}")
        notebook = None
        try:
            notebook = self._start_notebook_client(notebook_path)

            # Original logic to find index
            cell_index = self._notebook_cell_id_to_index(notebook, cell_id)
            if cell_index is None: # Check if index was found
                 # No need to stop notebook here, finally block handles it
                 raise ValueError(f"Cell with id {cell_id} is not found in {notebook_path}")

            # Original execution logic
            logger.info(f"Executing cell at index {cell_index}")
            notebook.execute_cell(cell_index, self.kernel)

            # Original output access logic
            ydoc = notebook._doc
            # Safe access to outputs
            if ydoc and hasattr(ydoc, '_ycells') and cell_index < len(ydoc._ycells):
                cell_data = ydoc._ycells[cell_index]
                outputs = cell_data.get("outputs", []) # Use .get for safety
            else:
                 logger.warning(f"Could not access outputs for cell index {cell_index} in {notebook_path}")
                 outputs = []


            # Original output extraction logic
            if len(outputs) == 0:
                cell_output_str = ""
            else:
                # Ensure parts are strings and join them
                output_parts = [str(self.extract_output(output)) for output in outputs]
                cell_output_str = "\n".join(output_parts)


            # Return the processed string output
            return cell_output_str
        except Exception as e:
            logging.info("execute_cell error: %s", e)

        finally:
            pass
            #self._stop_notebook_client(notebook)

    def get_tools(self) -> List[StructuredTool]:
        """Return a list of tools that this class provides.

        Returns:
            List[StructuredTool]: List of structured tools
        """
        return [
            StructuredTool.from_function(
                func=self.create_notebook,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.read_notebook,
                parse_docstring=True
            ),
            # StructuredTool.from_function(
            #     func=self.get_project_details,
            #     parse_docstring=True
            # ),
            StructuredTool.from_function(
                func=self.get_notebooks_overview,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.add_execute_code_cell,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.add_code_cell,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.add_markdown_cell,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.update_code_cell_by_cell_id,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.update_markdown_cell_by_id,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.execute_cell,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.get_current_notebook_name,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.get_current_notebook_path,
                parse_docstring=True
            ),
            StructuredTool.from_function(
                func=self.get_current_notebook_cell_id,
                parse_docstring=True
            ),            
        ]
