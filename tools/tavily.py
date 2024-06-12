from dotenv import load_dotenv
from typing import Optional, Type
from tools.utils import SearchInput, ToolResponse
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

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

  def __init__(self):
    super().__init__()

  def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ToolResponse:
    """
    Searches for information related to a specific entity using FirecrawlApp.
    """
    params = {"pageOptions": {"fetchPageContent": True}}
    result = ""
    try:
      tool = TavilySearchResults()
      search_result = tool.invoke({"query": query})
      result = str(search_result)
    except Exception as e:
      return ToolResponse(result="", context={"error": f"unable to search the {query}, error: {e}"})
    
    return ToolResponse(result=result, context={}) 
  