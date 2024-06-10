from typing import List
from langchain.pydantic_v1 import BaseModel, Field

class ToolResponse(BaseModel):
  result: str
  context: dict

class ScrapeInput(BaseModel):
  url: str = Field(description="""The URL to scrape.""")

class SearchInput(BaseModel):
  query: str = Field(description="""The search query.""")

class UpdateDataInput(BaseModel):
  """Input for the Update Data tool."""
  data_to_update: List[dict] = Field(description="""The data points found, which should follow the format [{"name": "xxx", "value": "yyy", "reference": "url"}]""")
