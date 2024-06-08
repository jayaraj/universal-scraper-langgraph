

from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from agent_nodes import AgentState, call_model, call_tool, optimise_messages, should_continue

def workflow(llm, model: str, tool_executor: ToolExecutor) -> CompiledGraph:

  workflow = StateGraph(AgentState)
  workflow.add_node("agent", partial(call_model, llm=llm))
  workflow.add_node("action", partial(call_tool, llm=llm, tool_executor=tool_executor))
  workflow.add_node("optimise", partial(optimise_messages, llm=llm, model=model))
  workflow.set_entry_point("agent")
  workflow.add_conditional_edges("agent", should_continue,
      {
          "continue": "action",
          "end": END
      }
  )
  workflow.add_edge('action', 'optimise')
  workflow.add_edge('optimise', 'agent')

  return workflow.compile()