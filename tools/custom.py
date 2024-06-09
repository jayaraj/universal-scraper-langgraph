from typing import List
from typing import Optional, Type
from tools.utils import ToolResponse
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun

class UpdateDataInput(BaseModel):
  data_to_update: List[dict] = Field(description="""
    The new data points found, which should follow the format:
      [{"name": "xxx", "value": "yyy", "reference": "url"}]
  """)

class UpdateDataTool(BaseTool):
  name = "update_data"
  description = """
    Update the state with new data points found.
    Args:
        data_to_update (List[dict]): The new data points found, which should follow the format 
          [{"name": "xxx", "value": "yyy", "reference": "url"}]
    Returns:
        ToolResponse: A confirmation message with the updated data, or an error message if the update fails.
  """
  args_schema: Type[BaseModel] = UpdateDataInput
  return_direct: Type[BaseModel] = ToolResponse
  data_points: List[dict] = []

  def __init__(self, data_points_to_search: List[str] = []):
    super().__init__()
    self.data_points =  [{"name": dp, "value": None, "reference": None} for dp in data_points_to_search]

  def _run(self, data_to_update: List[dict], run_manager: Optional[CallbackManagerForToolRun] = None) -> ToolResponse:
      """
      Update the state with new data points found.
      """
      for obj in data_to_update:
          for dp in self.data_points:
              if dp['name'] == obj['name'] and dp['value'] is None:
                  dp["value"] = obj["value"]
                  if "reference" in obj:
                      dp["reference"] = obj["reference"]
      return ToolResponse(result=f"updated data: {data_to_update}", context={})
  
  def get_data_points(self):
    return self.data_points