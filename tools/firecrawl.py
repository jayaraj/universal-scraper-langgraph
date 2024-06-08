
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.tools import tool
from tools.utils import ToolResponse

load_dotenv()

@tool("scrape", return_direct=True)
def scrape(url: str) -> ToolResponse:
  """
  Scrapes content from a specified URL using FirecrawlApp.
  Args:
      url (str): The URL to scrape.

  Returns:
      ToolResponse: The scraped content in markdown format, or an error message if scraping fails.
  """
  try:
    app = FirecrawlApp()
    scraped_data = app.scrape_url(url)
  except Exception as e:
    return ToolResponse(result="", context={"error": f"unable to scrape the URL: {url}, error: {e}"})

  return ToolResponse(result=str(scraped_data.get("markdown", "")), context={"url": url})

@tool("search", return_direct=True)
def search(query, entity_name, data_points_to_search) -> ToolResponse:
  """
  Searches for information related to a specific entity using FirecrawlApp.

  Args:
      query (str): The search query.
      entity_name (str): The name of the entity to search for.
      data_points_to_search (list): The list of data points to search for in the content.

  Returns:
      ToolResponse: The search results, or an error message if the search fails.
  """
  params = {"pageOptions": {"fetchPageContent": True}}
  try:
    app = FirecrawlApp()
    search_result = app.search(query, params=params)
    result = str(search_result)
  except Exception as e:
    return ToolResponse(result="", context={"error": f"unable to search the {query}, error: {e}"})
  
  return ToolResponse(result=str(result), context={"query": query, "entity_name": entity_name, "data_points_to_search": data_points_to_search})

@tool("update_data", return_direct=True)
def update_data (data_to_update) -> ToolResponse:
    """
    Update the state with new data points found.

    Args:
        data_to_update (List[dict]): The new data points found, which should follow the format 
        [{"name": "xxx", "value": "yyy", "reference": "url"}]

    Returns:
        ToolResponse: A confirmation message with the updated data, or an error message if the update fails.
    """
    return ToolResponse(result=f"updated data: {data_to_update}", context={"data_to_update": data_to_update})