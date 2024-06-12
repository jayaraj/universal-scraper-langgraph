from typing import List
from termcolor import colored
from dotenv import load_dotenv
from typing import Optional, Type
from firecrawl import FirecrawlApp
from tools.utils import ScrapeInput, SearchInput, ToolResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import CallbackManagerForToolRun

load_dotenv()

class ScrapeTool(BaseTool):
  name = "scrape"
  description = """
    Scrapes content from a specified URL using FirecrawlApp.
    Args:
        url (str): The URL to scrape.
    Returns:
        ToolResponse: The scraped content in markdown format, or an error message if scraping fails.
  """
  args_schema: Type[BaseModel] = ScrapeInput
  return_direct: bool = True
  links_already_scraped: list[str] = []

  def __init__(self):
    super().__init__()
    self.links_already_scraped =  []

  def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ToolResponse:
    """
    Scrapes content from a specified URL using FirecrawlApp.
    """

    ### Skippping scraping if the url is already scraped
    if url in self.links_already_scraped:
      return ToolResponse(result="", context={"error": f"this url: {url} is already scraped; please use different relevant url from the content scraped before."})

    try:
      app = FirecrawlApp()
      scraped_data = app.scrape_url(url)
    except Exception as e:
      return ToolResponse(result="", context={"error": f"unable to scrape the URL: {url}, error: {e}"})

    self.links_already_scraped.append(url)
    return ToolResponse(result=str(scraped_data.get("markdown", "")), context={})
  
  def get_links_already_scraped(self):
    return self.links_already_scraped

class SearchTool(BaseTool):
  name = "search"
  description = """
    Searches for information related to a specific entity using FirecrawlApp.

    Args:
        query (str): The search query.
        
    Returns:
        ToolResponse: The search results, or an error message if the search fails.
  """
  args_schema: Type[BaseModel] = SearchInput
  return_direct: bool = True
  llm: ChatOpenAI = None
  entity_name: str = ""
  data_points_to_search: List[str] = []

  def __init__(self, llm, entity_name: str, data_points_to_search: List[str] = []):
    super().__init__()
    self.llm = llm
    self.entity_name = entity_name
    self.data_points_to_search = data_points_to_search

  def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ToolResponse:
    """
    Searches for information related to a specific entity using FirecrawlApp.
    """
    params = {"pageOptions": {"fetchPageContent": True}}
    result = ""
    try:
      app = FirecrawlApp()
      search_result = app.search(query, params=params)
      result = str(search_result)
    except Exception as e:
      return ToolResponse(result="", context={"error": f"unable to search the {query}, error: {e}"})
    
    return self.extract_search_information(query, result)
  
  def extract_search_information(self, query, content: str) -> ToolResponse:
    """
    Extracts search information from the content based on the specified data points to search.

    Args:
        content (str): The content to extract information from.
    Returns:
        ToolResponse: The extracted search information, or an error message if the extraction fails.
    """

    message = HumanMessage(content=f"""
      Below are some search results from the internet about {query}:
      {content}
      -----
        
      Your goal is to find specific information about an entity called {self.entity_name} regarding {self.data_points_to_search}.

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
      response = self.llm.invoke([message])
      extracted_information = response.content
    except Exception as e:
      print(colored(f"error while extracting information: {e}", "red"))
      return ToolResponse(result="", context={"error": f"error while extracting information: {e}"})

    return ToolResponse(result=str(extracted_information), context={}) 
