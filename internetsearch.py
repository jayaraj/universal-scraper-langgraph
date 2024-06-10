from typing import Any, List
from graph import workflow
from tools.jinaai import ScrapeTool
from tools.tavily import SearchTool
from tools.custom import UpdateDataTool, update_data_definition
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

def internet_search(llm, model: str, entity_name: str, data_points_to_search: List[str]) -> (dict[str, Any] | Any):
  # search = SearchTool(llm, entity_name, data_points_to_search)
  scrape = ScrapeTool()
  update_data = UpdateDataTool(data_points_to_search)
  search = SearchTool()
  llm = llm.bind(tools=[convert_to_openai_tool(tool) for tool in [update_data_definition, search, scrape ]])
  tool_executor = ToolExecutor([update_data, search, scrape ])

  app = workflow(llm=llm, model=model, tool_executor=tool_executor)

  # system_message = SystemMessage(content="""
  #   You are a world class web researcher, you are great at finding information on the internet;
  #   You will keep scraping url based on information you received until information is found;

  #   You will try as hard as possible to search for all sorts of different query & source to find information; 
  #   You will make sure the information is accurate and up to date or latest information;
  #   if one search query didn't return any result, try another one; 
  #   You do not stop until all information are found, it is very important we find all information;
  #   Whenever you found required data points, use "update_data" function to save the data point, before proceeding to next query;
    
  #   You only answer questions based on results from scraper, do not make things up;
  #   You never ask user for inputs or permissions, you just do your job and provide the results;
  #   You NEVER make more than two tool_calls in a single turn;
  # """)

  system_message = SystemMessage(content="""
    You are a world-class web researcher and scraper. Your goal is to find comprehensive and up-to-date 
    information on a given entity by leveraging the following tools:
      1.search(query: str): Finds URLs and provides a brief description of their content.
	    2.scrape(url: str): Retrieves detailed content from the specified URL.
	    3.update_data(data_to_update: List[dict]): Saves the data points you have found,  where format is [{"name": "xxx", "value": "yyy", "reference": "url"}]
      , stricly update name from data_points_to_search.

    Instructions:
      1.Initiate Search:
	      *	When given an entity_name, start by using the search tool to find relevant URLs and descriptions.
	      * Prioritize finding the official website of the entity and scrape it first for comprehensive information.
	      * Only search for data points that are not yet found and are included in the provided data points to search list.
	      * Continue searching until all required data points are found to the best extent possible.
	    2.Scraping Process:
	      * For each URL obtained from the search results, use the scrape tool to extract detailed content.
        * Scrape one URL and update the extracted data using the update_data tool before proceeding to the next URL.
	      * If you encounter URLs within the scraped content that are relevant to the entity, scrape those URLs as well.
	      * Persist in scraping additional URLs if necessary to find the required data points.
	    3.Updating Data:
	      * Match the extracted data points with the data points specified in the data_points_to_search list.
        * data_to_update = [{"name": "xxx", "value": "yyy", "reference": "url"}] where name stricly part of data_points_to_search list.
	      * Ensure the data_to_update dicts correspond accurately to the data points specified in the data_points_to_search list.
	      * Verify the accuracy of each data point before updating.
	      * Immediately use the update_data tool with data_to_update containing the necessary information before proceeding to the next URL.
	    4.Efficiency:
	      * You will not scrape a URL more than once, even if the information is not found.
	      * Do not ask the user for inputs or permissions; proceed autonomously to gather the required information.
	      * Do not make more than two tool calls in a single turn.
	      * Stop the search if all required data points are found or if no data points are found in the search list.
	    5.Accuracy:
	      * Ensure that all information is accurate and up-to-date.
	      * Only answer questions based on results from the scraping process. Do not fabricate information.
  """)

  internet_search = HumanMessage(content=f"""
    entity_to_search : {entity_name}
    data_points_to_search: {data_points_to_search}

	  * Search continuously for the data points that are not found yet and are in the data points to search list.
	  * Stop if all data points are found.
	  * Stop if no data points are found in the search list.
	  * Do not search for data points that are already found.
    
    For example, if data_points_to_search = [“company_name”, “company_phone”, “company_email”];
    The update_data expects the update_to_data dicts from the data_points_to_search list.
  """)

  inputs = {"messages": [system_message, internet_search], 
            "data_points": [{"name": dp, "value": None, "reference": None} for dp in data_points_to_search],
            "links_already_scraped": []}
  config = {"recursion_limit": 100}
  app.invoke(inputs,config=config)

  return {"data_points": update_data.get_data_points()}