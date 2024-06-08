
from openai import BaseModel

class ToolResponse(BaseModel):
  result: str
  context: dict