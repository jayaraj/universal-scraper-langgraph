import requests
from typing import Optional, Type
from tools.utils import ScrapeInput, ToolResponse
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel
from langchain_core.callbacks import CallbackManagerForToolRun

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
    Scrapes content from a specified URL using jina AI.
    """
    ### Skippping scraping if the url is already scraped
    if url in self.links_already_scraped:
      return ToolResponse(result="", context={"error": f"this url: {url} is already scraped; please use different relevant url from the content scraped before."})

    try:
      response = requests.get("https://r.jina.ai/" + url)
    except Exception as e:
      return ToolResponse(result="", context={"error": f"unable to scrape the URL: {url}, error: {e}"})

    self.links_already_scraped.append(url)
    return ToolResponse(result=str(response.text), context={})
  
  def get_links_already_scraped(self):
    return self.links_already_scraped
