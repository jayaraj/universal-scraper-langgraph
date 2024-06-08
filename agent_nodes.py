import operator
import tiktoken
from termcolor import colored
from langgraph.prebuilt import ToolInvocation
from typing import List, TypedDict, Annotated, Sequence
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage

class AgentState(TypedDict):
   messages: Sequence[BaseMessage]
   data_points: List[dict]
   links_already_scraped: Annotated[List[str], operator.add]

def call_model(state, llm):
  messages = state['messages']
  response = None
  try:
    response = llm.invoke(messages)
  except Exception as e:
    print(colored(f"error while processing messages: {e}", "red"))
  return {"messages": messages + [response]}

def should_continue(state):
  messages = state['messages']
  last_message = messages[-1]
  if last_message is None or "tool_calls" not in last_message.additional_kwargs:
    return "end"
  else:
    return "continue"
  
def call_tool(state, llm, tool_executor):
  messages = state['messages']
  last_message = messages[-1]
  data_points = state['data_points']
  links_already_scraped = []
  tool_messages = []
  print(f"No of tool calls: {len(last_message.tool_calls)}")
  for tool_call in last_message.tool_calls:
    action = ToolInvocation(
      tool=tool_call["name"],
      tool_input=tool_call["args"]
    )
    print(f"""The agent action is {action} and the tool call id: {tool_call["id"]}""")
    ### Skippping scraping if the url is already scraped
    if action.tool == "scrape":
      if state['links_already_scraped'] is not None and "url" in action.tool_input and action.tool_input["url"] in state['links_already_scraped']:
          tool_messages.append(ToolMessage(content=f"""
            this url: {action.tool_input["url"]} is already scraped;
            please use different relevant url from the content scraped before.""", 
            name=action.tool, 
            tool_call_id=tool_call["id"]))
          continue
    response = tool_executor.invoke(action)
    content = response.result
    if "error" in response.context:
      print(colored(f"""error: {response.context["error"]}""", "red"))
      content = response.context["error"]

    ### Update the data points with the new data points found
    if action.tool == "update_data" and "data_to_update" in response.context:
      for obj in response.context["data_to_update"]:
          for dp in data_points:
              if dp['name'] == obj['name'] and dp['value'] is None:
                  dp["value"] = obj["value"]
                  if "reference" in obj:
                      dp["reference"] = obj["reference"]
    
    if action.tool == "search" and "query" in response.context:
      if "entity_name" in response.context and "data_points_to_search" in response.context:
          content = extract_search_information(llm, content, response.context["query"], response.context["entity_name"], response.context["data_points_to_search"])

    tool_messages.append(ToolMessage(content=str(content), name=action.tool, tool_call_id=tool_call["id"]))

    ### Update the links already scraped
    if action.tool == "scrape" and "url" in response.context:
      if state['links_already_scraped'] is None or response.context["url"] not in state['links_already_scraped']:
          links_already_scraped = [response.context["url"]]

  return {"messages": messages + tool_messages, "data_points": data_points, "links_already_scraped": links_already_scraped}

def extract_search_information(llm, content: str, query: str, entity_name: str, data_points_to_search: list) -> str:
    """
    Extracts search information from the content based on the specified data points to search.

    Args:
        content (str): The content to extract information from.
        entity_name (str): The name of the entity to search for.
        data_points_to_search (list): The list of data points to search for in the content.

    Returns:
        str: The extracted search information, or an error message if the extraction fails.
    """

    message = HumanMessage(content=f"""
      Below are some search results from the internet about {query}:
      {content}
      -----
        
      Your goal is to find specific information about an entity called {entity_name} regarding {data_points_to_search}.

      Please extract information from the search results above in the following JSON format:
      {{
          "related_urls_to_scrape_further": ["url1", "url2", "url3"],
          "info_found": [
              {{
                  "research_item": "xxxx",
                  "reference": "url"
              }},
              {{
                  "research_item": "yyyy",
                  "reference": "url"
              }}
              ...
          ]
      }}

      Where "research_item" is the actual research item name you are looking for.

      Only return research items that you actually found.
      If no research item information is found from the content provided, just don't return any research item.

      Extracted JSON:
      {{
          "related_urls_to_scrape_further": [],
          "info_found": []
      }}
    """)

    extracted_information = ""
    try:
      response = llm.invoke([message])
      extracted_information = response.content
      print(colored(f"summary: {extracted_information}", "green"))
    except Exception as e:
      print(colored(f"error while extracting information: {e}", "red"))
      return content

    return extracted_information

def optimise_messages(state, llm, model):
  messages = state['messages']
  system_prompt = messages[0].content
  encoding = tiktoken.encoding_for_model(model)

  token_count_messages = len(encoding.encode(str(messages)))
  print(f"token count of messages: {token_count_messages} for {len(messages)} messages")

  if len(messages) > 10 or token_count_messages > 5000:
    latest_messages = messages [-5:]
    
    if isinstance(latest_messages[0], ToolMessage):
      latest_messages = latest_messages[1:]

    index = messages.index(latest_messages[0])
    early_messages = messages[:index]

    token_count_latest_messages = len(encoding.encode(str(latest_messages) ))
    print(f"token count of latest messages: {token_count_latest_messages}  for {len(latest_messages)} latest messages")

    message = HumanMessage(content=f"""
      Conversation History:
      {early_messages}
      -----
      
      Above is the conversation history between the user and the AI, including actions the AI has already taken.
      Please summarize the past actions taken so far, highlight any key information learned, and mention tasks that have been completed.
      Remove any redundant information and keep the summary concise. remove unnessary scrapped content from the summary.

      SUMMARY:""")
    summary = ""
    try:
      response = llm.invoke([message])
      summary = response.content
    except Exception as e:
      print(colored(f"error while optimising messages: {e}", "red"))
      return {"messages": messages}
    
    system_message = SystemMessage(content=f"""
      {system_prompt};
      -------
      Here is a summary of past actions taken so far:
      {summary}
      """)
    
    optimised_messages = [system_message] + latest_messages
    print(f"token count of optimised messages: {len(encoding.encode(str(optimised_messages)))} for {len(optimised_messages)} optimised messages")
    return {"messages": optimised_messages}

  return {"messages": messages}

    
      
    
      
    
