import requests
from langchain.tools import tool
from tools.utils import ToolResponse

@tool("scrape", return_direct=True)
def scrape(url: str) -> ToolResponse:
  """
  Scrapes content from a specified URL using jina AI.
  Args:
      url (str): The URL to scrape.

  Returns:
      ToolResponse: The scraped content in markdown format, or an error message if scraping fails.
  """
  try:
    response = requests.get("https://r.jina.ai/" + url)
  except Exception as e:
    return ToolResponse(result="", context={"error": f"unable to scrape the URL: {url}, error: {e}"})

  return ToolResponse(result=str(response.text), context={"url": url})