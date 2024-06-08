from typing import Any, List
from graph import workflow
from tools.jinaai import scrape
from tools.firecrawl import update_data
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

def website_scrap(llm, model: str, website: str, entity_name: str, data_points_to_search: List[str], links_already_scraped: List[str]) -> (dict[str, Any] | Any):
    
  tools = [scrape, update_data]
  llm = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
  tool_executor = ToolExecutor(tools)

  app = workflow(llm=llm, model=model, tool_executor=tool_executor)

  system_message = SystemMessage(content="""
    You are a world class web scraper, you are great at finding information on urls;
    You will not scrape urls more than once even if information is not found;
    You will not use any other tools other than the scrape and update_data;
    
    If you find relevant urls of the entity from the content, scrape all those urls if information is not found;                            
    Whenever you found certain data point, use "update_data" function to save the data point;
                          
    You only answer questions based on results from scrape tool, do not make things up;
    Scraper will return empty string if your are trying to scrape a url that is already scraped;
    
    You NEVER ask user for inputs or permissions, just go ahead do the best thing possible without
    asking for permission or guidance from user;
                          
    You NEVER make more than two tool_calls in a single turn;
  """)

  web_scrape = HumanMessage(content=f"""
    Website to scrape: {website}

    Entity name: {entity_name}

    Data points to search: {data_points_to_search}

    Search only the data points that are not found yet and are in the data points to search list.
    Stop if all data points are found.
    Stop if no data points are found in search list.
    Need not search for data points that are already found.
  """)

  inputs = {"messages": [system_message, web_scrape], 
            "data_points": [{"name": dp, "value": None, "reference": None} for dp in data_points_to_search],
            "links_already_scraped": links_already_scraped}
  config = {"recursion_limit": 30}

  return app.invoke(inputs,config=config)


  