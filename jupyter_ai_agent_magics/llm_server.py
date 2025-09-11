from typing import Dict, List, Optional, Any, Literal

from langchain.prompts import PromptTemplate
#from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatAnthropic
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.prebuilt import create_react_agent

from langchain.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent  # Import from langgraph
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.tool import ToolMessage

from jupyter_ai_agent_magics import model_factory
import ipywidgets as widgets
import re
import html

import json
import ast

from IPython.display import display, HTML, clear_output
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter

from collections.abc import Iterable

from ansi2html import Ansi2HTMLConverter

# ANSI escape codes for styling
class ANSI:
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[96m"   # Bright cyan
    GREEN = "\033[92m"  # Bright green
    YELLOW = "\033[93m" # Bright yellow
    BLUE = "\033[94m"   # Bright blue

def pretty_json(json_obj):
    """Format JSON for display."""
    return json.dumps(json_obj, indent=2, sort_keys=True)
    

def pretty_json_html(json_obj):
    """Convert a JSON object to a styled HTML string using Pygments."""
    # Convert JSON to a pretty-printed string
    json_str = json.dumps(json_obj, indent=2, sort_keys=True)
    
    # Highlight JSON using Pygments
    highlighted = highlight(json_str, JsonLexer(), HtmlFormatter(style='default'))
    
    return highlighted

def append_to_widget(output_widget, content, html_escape=True):
    """Append content to the output widget and return updated content."""
    if html_escape:
        if output_widget.value and content:
            if output_widget.value[-1] == "\\" and content[0] == "n":
                output_widget.value = output_widget.value[:-1]
                content = "<br>" + content[1:]
            if output_widget.value[-1] == "\\" and content[0] == '"':
                output_widget.value = output_widget.value[:-1]
        html_content = content.replace("\n", "<br>")
        html_content = html_content.replace('\\"', '"')
        output_widget.value += html_content
    else:
        output_widget.value += content

class LLMServer:
    def __init__(
        self, 
        model_provider: Literal["anthropic", "vertex"] = "anthropic",
        model_name: str = "claude-3-opus-20240229", 
        api_key: Optional[str] = None, 
        project_id: Optional[str] = None,
        location: str = "us-central1",
        temperature: Optional[float] = None,
        tools: List = []
    ):
        self._last_tool_call_id = None
        self._model = model_factory.create_model(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key,
            project_id=project_id,
            location=location,
            temperature=temperature)

        self._history = []
    
        system_prompt = (
            """You are an AI assistant integrated with JupyterLab. 
            Your role is to help users with notebook operations and data analysis tasks.
            If you need a human input, call one of the provided ask_human... tools.
            """)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
    
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", "{input}"),
        #         # Placeholders fill up a **list** of messages
        #         ("placeholder", "{agent_scratchpad}"),
        #     ]
        # )
        self._memory = MemorySaver()
        # https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/
        #self._agent = create_tool_calling_agent(self._model, tools, prompt=prompt)
        self._agent = create_react_agent(self._model, tools, prompt=prompt, checkpointer=self._memory)
        #self._agent_executor = AgentExecutor(agent=self._agent, tools=tools, checkpointer=self._memory)


    #TODO: handle user interactions:
    #for message in stream:
    #print(message)
    #if message.get("type") == "tool_call" and message["tool_call"]["name"] == "human_assistance":
    #    print("Human assistance requested — interrupt triggered")
    #    # Here, pause for user input or route accordingly
    # continue:
    # human_response = (
    #     "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    #     " It's much more reliable and extensible than simple autonomous agents."
    # )
    
    # human_command = Command(resume={"data": human_response})
    
    # events = graph.stream(human_command, config, stream_mode="values")
    # for event in events:
    #     if "messages" in event:
    #         event["messages"][-1].pretty_print()    
        
    def process_message_debug(self, query: str, thread_id: Optional[str] = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self._agent.stream(
            {"messages": [("human", query)]}, stream_mode="messages", config=config
        ):

            try:
                print(" \n" + str(chunk) + " \n" )
            except Exception as e:
                print("exception while serializing chunk:", e)
                breakpoint()
                print("metadata:", chunk[1])

    def ask_human_debug(self, query, thread_id: Optional[str] = "1"):
        def on_submit_clicked(_):
            human_response = text_input.value.strip()
            query = "human answer:" + human_response
            print("response :", human_response)
            self.process_message_debug_human_interaction(query, thread_id=thread_id)
        flow_box = widgets.VBox()
        display(flow_box)
        prompt_label  = widgets.Label(value=f"query")
        text_input    = widgets.Text(placeholder="Type your answer here...",
                                     layout=widgets.Layout(width="300px"))
        submit_button = widgets.Button(description="Submit")
        submit_button.on_click(on_submit_clicked)
        flow_box.children = tuple(
            list(flow_box.children) + [prompt_label, text_input, submit_button]
        )
    
    def process_message_debug_human_interaction(self, query: str, thread_id: Optional[str] = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self._agent.stream(
            {"messages": [("human", query)]}, stream_mode="messages", config=config
        ):
            chunk_content, metadata = chunk
            if isinstance(chunk_content, ToolMessage):
                print(">>> ToolMessage, chunk_content", chunk_content)
                if chunk_content.name == "ask_human":
                    print(">>> ask_human was invoked")
                    self.ask_human_debug("test query", thread_id=thread_id)
                
            try:
                print(" \n" + str(chunk) + " \n" )
            except Exception as e:
                print("exception while serializing chunk:", e)
                breakpoint()
                print("metadata:", chunk[1])


    def process_message(self, query: str, thread_id: Optional[str] = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        
        # Print the query with bold and cyan
        print(f"{ANSI.BOLD}Processing query:{ANSI.RESET} {query}")
        print("-" * 50)  # Separator line
        
        seen_text = set()
        pending_tool_results = []
        current_tool_name = None
        tool_args_buffer = ""
        
        for chunk in self._agent.stream(
            {"messages": [("human", query)]}, stream_mode="messages", config=config
        ):
            chunk_content, metadata = chunk
            
            if isinstance(chunk_content, AIMessageChunk):
                if isinstance(chunk_content.content, str):
                    # Google
                    # AIMessageChunk(content='Hello' ...
                    if isinstance(chunk_content.content, str) and chunk_content.content:
                        #append_to_widget(output_widget, "\n --google stuff chunk_content.content -- \n")    
                        text = chunk_content.content
                        print(text)
                    # AIMessageChunk(content='', ...  tool_calls=[{'name': 'create_notebook', 'args': {'path': 'fibonachi.ipynb'},...
                    if chunk_content.tool_calls:
                        #append_to_widget(output_widget, "\n --google stuff chunk_content.tool_calls -- \n") 
                        for tool_call in chunk_content.tool_calls:
                            tool_name = tool_call['name']
                            print(f"\n\n{ANSI.BOLD}Calling tool:{ANSI.RESET} {tool_name}")
                else:
                    # Anthropic
                    for content in chunk_content.content:
                        # Handle streaming text
                        if 'text' in content:
                            text = content['text']
                            if text not in seen_text:
                                seen_text.add(text)
                                print(text, end="")  # Inline text
                        
                        # Handle tool call declaration
                        if 'type' in content and content['type'] == 'tool_use' and 'name' in content:
                            tool_name = content['name']
                            print(f"\n\n{ANSI.BOLD}Calling tool:{ANSI.RESET} {tool_name}")
                        
                        # Handle streaming tool arguments
                        if 'type' in content and content['type'] == 'tool_use' and 'partial_json' in content:
                            tool_args_buffer += content['partial_json']
                            try:
                                tool_args = json.loads(tool_args_buffer)
                                print(f"{ANSI.BOLD}Arguments:{ANSI.RESET}")
                                print(json.dumps(tool_args, indent=2))
                                tool_args_buffer = ""
                            except json.JSONDecodeError:
                                pass
            
            # Handle ToolMessage (tool execution result)
            elif isinstance(chunk_content, ToolMessage):
                tool_name = chunk_content.name
                tool_output = chunk_content.content.strip()

                if current_tool_name and current_tool_name != tool_name:
                    if pending_tool_results:
                        print(f"\n{ANSI.BOLD}{ANSI.BLUE}Tool Result ({current_tool_name}):{ANSI.RESET}")
                        print(" ".join(pending_tool_results))
                    pending_tool_results = []
                
                current_tool_name = tool_name
                pending_tool_results.append(tool_output)
        
        # Flush remaining tool results
        if pending_tool_results and current_tool_name:
            print(f"\n{ANSI.BOLD}{ANSI.BLUE}Tool Result ({current_tool_name}):{ANSI.RESET}")
            print(" ".join(pending_tool_results))

    def adjust_agent_state_with_human_response(self, thread_id: str, human_response: str):
        # Get the current state
        
        config = {"configurable": {"thread_id": thread_id}}
        current_state = self._agent.get_state(config)
        
        if not current_state or not current_state.values.get("messages"):
            return
        messages = current_state.values["messages"]
        if isinstance(messages[-1], AIMessage) and not messages[-1].content:
            messages[-1].content = [{"text": " ", "type": "text"}]
        if isinstance(messages[-2], AIMessage) and not messages[-2].content:
            messages[-2].content = [{"text": " ", "type": "text"}]
        if (isinstance(messages[-1], ToolMessage) and 
            messages[-1].content == 'stop execution, response will be provided in next query' and
            messages[-1].name and
            messages[-1].name.startswith("ask_human")
           ):
            messages[-1].content = human_response
    
        # Update the state with the filtered messages
        self._agent.update_state(config, values={"messages": messages})

    def get_last_tool_call_args(self, thread_id: str) -> str | None:
        config = {"configurable": {"thread_id": thread_id}}
        current_state = self._agent.get_state(config)
        
        if not current_state or not current_state.values.get("messages"):
            return
        messages = current_state.values["messages"]
        for message in reversed(messages):
            if (isinstance(message, AIMessage) and
                message.tool_calls):
                return message.tool_calls[0]["args"]
        return None


    def process_message_html(self, query: str, thread_id: Optional[str] = "1", output_widget = None, flow_box = None):
        def ask_human(tool_query, thread_id: Optional[str] = "1"):
            def on_submit_clicked(_):
                flow_box.children = []
                human_response = text_input.value.strip()
                query = "Human answer:" + human_response
                try:
                    self.adjust_agent_state_with_human_response(thread_id, human_response)
                    append_to_widget(
                        output_widget,
                        '<hr>'
                    )                    
                    self.process_message_html(
                        query, 
                        thread_id=thread_id, 
                        output_widget=output_widget, 
                        flow_box=flow_box)
                except Exception as e:
                    append_to_widget(
                        output_widget,
                        "!!!!!!! error !!!!!!\n"
                    )                
                    append_to_widget(
                        output_widget,
                        str(e)
                    )
                    breakpoint()
            # flow_box = widgets.VBox()
            # display(flow_box)
            prompt_label  = widgets.Label(value=tool_query)
            text_input    = widgets.Text(placeholder="Type your answer here...",
                                         layout=widgets.Layout(width="300px"))
            text_input.on_submit(on_submit_clicked)
            submit_button = widgets.Button(description="Submit")
            submit_button.on_click(on_submit_clicked)
            flow_box.children = tuple(
                list(flow_box.children) + [prompt_label, text_input, submit_button]
            )

        def ask_human_yes_or_no(tool_query, thread_id: Optional[str] = "1"):
            def on_clicked(human_response):
                flow_box.children = []
                query = "Human answer:" + human_response
                try:
                    self.adjust_agent_state_with_human_response(thread_id, human_response)
                    append_to_widget(
                        output_widget,
                        '<hr>'
                    )                    
                    self.process_message_html(
                        query, 
                        thread_id=thread_id, 
                        output_widget=output_widget, 
                        flow_box=flow_box)
                except Exception as e:
                    append_to_widget(
                        output_widget,
                        "!!!!!!! error !!!!!!"
                    )                
                    append_to_widget(
                        output_widget,
                        str(e)
                    )
            prompt_label  = widgets.Label(value=tool_query)
            yes_button = widgets.Button(description="Yes")
            yes_button.on_click(lambda _:on_clicked("yes"))
            no_button = widgets.Button(description="No")
            no_button.on_click(lambda _:on_clicked("no"))
            flow_box.children = tuple(
                list(flow_box.children) + [prompt_label, yes_button, no_button]
            )

        def ask_human_to_choose_option(tool_query, options: list[str], thread_id: Optional[str] = "1"):
            def on_clicked(human_response):
                flow_box.children = []
                query = "Human answer:" + human_response
                try:
                    self.adjust_agent_state_with_human_response(thread_id, human_response)
                    append_to_widget(
                        output_widget,
                        '<hr>'
                    )                    
                    self.process_message_html(
                        query, 
                        thread_id=thread_id, 
                        output_widget=output_widget, 
                        flow_box=flow_box)
                except Exception as e:
                    append_to_widget(
                        output_widget,
                        "!!!!!!! error !!!!!!"
                    )                
                    append_to_widget(
                        output_widget,
                        str(e)
                    )
            prompt_label  = widgets.Label(value=tool_query)
            buttons = []
            for option in options:
                button = widgets.Button(description=option)
                button.on_click(lambda _:on_clicked(option))
                buttons.append(button)
            flow_box.children = tuple(
                list(flow_box.children) + [prompt_label] + buttons
            )
        
        config = {"configurable": {"thread_id": thread_id}}

        if not output_widget:
            output_widget = widgets.HTML()
            display(output_widget)
            flow_box = widgets.VBox()
            display(flow_box)

        seen_text = set()
        pending_tool_results = []
        current_tool_name = None
        tool_args_buffer = ""  # Buffer for streaming tool arguments
        
        append_to_widget(
            output_widget,
            f'<b>Processing query:</b> <span>{query}</span><hr>'
        )
        
        for chunk in self._agent.stream(
            {"messages": [("human", query)]}, stream_mode="messages", config=config
        ):
            chunk_content, metadata = chunk
            self._history.append(chunk_content)
            # append_to_widget(
            #     output_widget,
            #     "\n --start-- \n" + str(type(chunk_content)) + "\n" + str(chunk_content) + "\n --end-- \n"
            # )              
            if isinstance(chunk_content, AIMessageChunk):
                if isinstance(chunk_content.content, str):
                    #append_to_widget(output_widget, "\n --google stuff -- \n")     
                    # Google
                    # AIMessageChunk(content='Hello' ...
                    if isinstance(chunk_content.content, str) and chunk_content.content:
                        #append_to_widget(output_widget, "\n --google stuff chunk_content.content -- \n")    
                        text = chunk_content.content
                        append_to_widget(
                            output_widget,
                            text
                        )
                    # AIMessageChunk(content='', ...  tool_calls=[{'name': 'create_notebook', 'args': {'path': 'fibonachi.ipynb'},...
                    if chunk_content.tool_calls:
                        #append_to_widget(output_widget, "\n --google stuff chunk_content.tool_calls -- \n") 
                        for tool_call in chunk_content.tool_calls:
                            tool_name = tool_call['name']
                            if tool_name:
                                html = f'<br><b>Calling tool:</b> <pre>{tool_name}</pre>'
                                append_to_widget(
                                    output_widget,
                                    html
                                )
                    # OpenAI
                    # TODO: support rendering too args through chunk_content.tool_call_chunks
                else:
                    # Anthropic
                    for content in chunk_content.content:
                        # Normal case, streaming text
                        # AIMessageChunk(content=[{'text': ' the Fibonacci sequence an', 'type': 'text', 'index': 0}]    
                        if 'text' in content:
                            text = content['text']
                            append_to_widget(
                                output_widget,
                                text
                            )
                        # declaring a tool call:
                        #AIMessageChunk(content=[{'id': 'toolu_vrtx_01DpGhbyTWWKExPCqy5MYuLp', 'input': {}, 'name': 'add_execute_code_cell', 'type': 'tool_use', 'index': 1}                    
                        if 'type' in content and content['type'] == 'tool_use' and 'name' in content:
                            tool_name = content['name']
                            html = f'<br><b>Calling tool:</b> <pre>{tool_name}</pre>'
                            append_to_widget(
                                output_widget,
                                html
                            )
                            # debug:
                            # append_to_widget(
                            #     output_widget,
                            #     str(content)
                            # )
                        # streaming tool args
                        # partial_json and arg and tool_call_chunks.args have same content
                        #AIMessageChunk(content=[{'partial_json': '{"', 'type': 'tool_use', 'index': 1}], additional_kwargs={}, response_metadata={}, id='run-1fa70610-9fa2-4ec5-92e2-8afc57a24146', tool_calls=[{'name': '', 'args': {}, 'id': None, 'type': 'tool_call'}], tool_call_chunks=[{'name': None, 'args': '{"', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
                        if 'type' in content and content['type'] == 'tool_use' and 'partial_json' in content:
                            text = content['partial_json'].replace('\\n', '<br>')
                            append_to_widget(
                                output_widget,
                                text
                            )
            
            # Handle ToolMessage (tool execution result)
            # ToolMessage(content='First 10 numbers in the Fibonacci sequence:\n[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n', name='add_execute_code_cell', id='698fabfd-a6cf-4013-8ecc-6ce163d3c590', tool_call_id='toolu_vrtx_01DpGhbyTWWKExPCqy5MYuLp')
            elif isinstance(chunk_content, ToolMessage):
                tool_name = chunk_content.name
                tool_output = chunk_content.content.strip()
                html_escape = False

                if tool_name.startswith("ask_human"):
                    tool_args = self.get_last_tool_call_args(thread_id)
                    if tool_name == "ask_human" and "query" in tool_args:
                        ask_human(tool_args["query"], thread_id=thread_id)
                    if tool_name == "ask_human_yes_or_no" and "query" in tool_args:
                        ask_human_yes_or_no(tool_args["query"], thread_id=thread_id)
                    if (tool_name == "ask_human_to_choose_option" and 
                        "query" in tool_args and
                        "options" in tool_args
                       ):
                        ask_human_to_choose_option(tool_args["query"], tool_args["options"], thread_id=thread_id)       
                # Check if output is in ANSI format which is not JSON
                if "['\\x1b[" in tool_output and tool_output.endswith("]"):
                    try:
                        index = tool_output.index("['\\x1b[")
                        formatted_output = tool_output[:index].rstrip("\n")
                        ansi_text = "\n".join(ast.literal_eval(tool_output[index:]))
                        converter = Ansi2HTMLConverter()
                        formatted_output += converter.convert(ansi_text, full=True)                        
                    except SyntaxError:
                        html_escape = True
                        formatted_output = f'<pre>{tool_output}</pre>'
                else:
                    # Check if output is JSON
                    try:
                        json_output = json.loads(tool_output)
                        formatted_output = pretty_json_html(json_output)
                    except json.JSONDecodeError:
                        html_escape = True
                        formatted_output = f'<pre>{tool_output}</pre>'
                
                # Display tool result
                tool_result_html = (
                    f'<br><b>Tool Result ({tool_name}):</b><br>' + formatted_output
                )
                append_to_widget(
                    output_widget,
                    tool_result_html,
                    html_escape=html_escape
                )   
