import tiktoken
from termcolor import colored
from langgraph.prebuilt import ToolInvocation
from typing import TypedDict, Sequence
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage

class AgentState(TypedDict):
   messages: Sequence[BaseMessage]

def call_model(state, llm):
  messages = state['messages']
  response = None
  try:
    response = llm.invoke(messages)
  except Exception as e:
    print(colored(f"error while processing messages: {e}", "red"))
  return {"messages": messages + [response]}

def should_continue(state):
  messages = state['messages']
  last_message = messages[-1]
  if last_message is None or "tool_calls" not in last_message.additional_kwargs:
    return "end"
  else:
    return "continue"
  
def call_tool(state, tool_executor):
  messages = state['messages']
  last_message = messages[-1]
  tool_messages = []
  for tool_call in last_message.tool_calls:
    action = ToolInvocation(
      tool=tool_call["name"],
      tool_input=tool_call["args"]
    )
    print(f"""The agent action is {action} and the tool call id: {tool_call["id"]}""")
    response = tool_executor.invoke(action)
    content = response.result
    if "error" in response.context:
      print(colored(f"""error: {response.context["error"]}""", "red"))
      content = f"""error: {response.context["error"]}"""

    tool_messages.append(ToolMessage(content=str(content), name=action.tool, tool_call_id=tool_call["id"]))

  return {"messages": messages + tool_messages}

def optimise_messages(state, llm, model):
  messages = state['messages']
  system_prompt = messages[0].content
  encoding = tiktoken.encoding_for_model(model)

  token_count_messages = len(encoding.encode(str(messages)))
  print(f"token count of messages: {token_count_messages} for {len(messages)} messages")

  if len(messages) > 10 or token_count_messages > 5000:
    latest_messages = messages [-5:]
    
    for message in latest_messages:
      if isinstance(message, ToolMessage):
        latest_messages = latest_messages[1:]
      else:
        break

    index = messages.index(latest_messages[0])
    early_messages = messages[:index]

    token_count_latest_messages = len(encoding.encode(str(latest_messages) ))
    print(f"token count of latest messages: {token_count_latest_messages}  for {len(latest_messages)} latest messages")

    message = HumanMessage(content=f"""
      Conversation History:
      {early_messages}
      -----
      
      Above is the conversation history between the user and the AI, including actions the AI has already taken.
      Please summarize the past actions taken so far, highlight any key information learned, and mention tasks that have been completed.
      Remove any redundant information and keep the summary concise. remove unnessary scrapped content from the summary.

      SUMMARY:""")
    summary = ""
    try:
      response = llm.invoke([message])
      summary = response.content
      print(colored(f"summary: {summary}", "green"))
    except Exception as e:
      print(colored(f"error while optimising messages: {e}", "red"))
      return {"messages": messages}
    
    system_message = SystemMessage(content=f"""
      {system_prompt};
      -------
      Here is a summary of past actions taken so far:
      {summary}
      """)
    
    optimised_messages = [system_message] + latest_messages
    print(f"token count of optimised messages: {len(encoding.encode(str(optimised_messages)))} for {len(optimised_messages)} optimised messages")
    return {"messages": optimised_messages}

  return {"messages": messages}

    
      
    
      
    
