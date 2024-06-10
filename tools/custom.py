from typing import List
from langchain.tools import tool
from typing import Optional, Type
from tools.utils import ToolResponse, UpdateDataInput
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel 
from langchain_core.callbacks import CallbackManagerForToolRun

class UpdateDataTool(BaseTool):
  """Tool that updates state with new data points found."""
  name = "update_data"
  description = """
  Update the state with new data points found.

  Args:
      data_to_update (List[dict]): The new data points found, which should follow the format 
      [{"name": "xxx", "value": "yyy", "reference": "url"}]

  Returns:
      ToolResponse: A confirmation message with the updated data, or an error message if the update fails.
  """
  args_schema: Optional[Type[BaseModel]] = UpdateDataInput
  infer_schema: bool = True
  return_direct: bool = True
  data_points: List[dict] = []

  def __init__(self, data_points_to_search: List[str] = []):
    super().__init__()
    self.data_points =  [{"name": dp, "value": None, "reference": None} for dp in data_points_to_search]

  def _run(self, data_to_update: List[dict], run_manager: Optional[CallbackManagerForToolRun] = None) -> ToolResponse:
      """
      Update the state with new data points found.
      """
      for obj in data_to_update:
          if obj["name"] not in [dp["name"] for dp in self.data_points]:
             return ToolResponse(result="", context={"error": f"""{obj["name"]}  is not part of data_points_to_search : {[dp["name"] for dp in self.data_points]}, Please update with correct data points"""})
          for dp in self.data_points:
              if dp['name'] == obj['name'] and dp['value'] is None:
                  dp["value"] = obj["value"]
                  if "reference" in obj:
                      dp["reference"] = obj["reference"]
      return ToolResponse(result=f"updated data: {data_to_update}", context={})
  
  def get_data_points(self):
    return self.data_points
  

@tool("update_data", return_direct=True)
def update_data_definition (data_to_update) -> ToolResponse:
  """
  Update the state with new data points found.

  Args:
      data_to_update (List[dict]): The new data points found, which should follow the format 
      [{"name": "xxx", "value": "yyy", "reference": "url"}]

  Returns:
      ToolResponse: A confirmation message with the updated data, or an error message if the update fails.
  """
  return ToolResponse(result=f"updated data: {data_to_update}", context={"data_to_update": data_to_update})