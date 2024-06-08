from typing import Any, List
from graph import workflow
from tools.firecrawl import update_data, search
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

def internet_search(llm, model: str, entity_name: str, data_points_to_search: List[str]) -> (dict[str, Any] | Any):
    
  tools = [search, update_data]
  llm = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
  tool_executor = ToolExecutor(tools)

  app = workflow(llm=llm, model=model, tool_executor=tool_executor)

  system_message = SystemMessage(content="""
    You are a world class web researcher, you are great at finding information on the internet;
    You will keep scraping url based on information you received until information is found;

    You will try as hard as possible to search for all sorts of different query & source to find information; 
    if one search query didn't return any result, try another one; 
    You do not stop until all information are found, it is very important we find all information, I will give you $100, 000 tip if you find all information;
    Whenever you found certain data point, use "update_data" function to save the data point;
    
    You only answer questions based on results from scraper, do not make things up;
    You never ask user for inputs or permissions, you just do your job and provide the results;
    You NEVER make more than two tool_calls in a single turn;
  """)

  internet_search = HumanMessage(content=f"""
    Entity to search: {entity_name}
    Data points to search: {data_points_to_search}
    
    Search only the data points that are not found yet and are in the data points to search list.
    Stop if all data points are found.
    Stop if no data points are found in search list.
    Need not search for data points that are already found.
  """)

  inputs = {"messages": [system_message, internet_search], 
            "data_points": [{"name": dp, "value": None, "reference": None} for dp in data_points_to_search],
            "links_already_scraped": []}
  config = {"recursion_limit": 30}

  return app.invoke(inputs,config=config)