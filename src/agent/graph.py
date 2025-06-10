import operator
from typing import TypedDict, Annotated, Union, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_tool
import functools
import logging

# Local imports
from src.prompts.agent_prompts import SYSTEM_PROMPT_LANGGRAPH
from src.config.app_config import AGENT_MAX_ITERATIONS, TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION, MAX_TOOL_RETRIES
from src.agent.summarizer_graph import create_summarizer_graph

logger = logging.getLogger(__name__)

# Agent State Definition
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    agent_outcome: Optional[Union[List[AgentAction], AgentFinish]]
    retrieved_tool_context: Optional[str] = None
    summarized_tool_context: Optional[str] = None
    summarization_error: Optional[str] = None
    current_tool_action: Optional[AgentAction] = None
    current_tool_retries: int = 0
    last_tool_error: Optional[str] = None

# Module-level counter for iterations within a single graph invocation.
# Reset each time create_agent_graph is called.
_iteration_count_local = 0

def create_agent_graph(llm: BaseChatModel, tools_list: List):
    global _iteration_count_local
    _iteration_count_local = 0 # Reset counter for new graph creation

    llm_with_tools = llm.bind_tools(tools_list)

    # Construct tools_description string
    if tools_list:
        tools_description = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools_list]
        )
    else:
        tools_description = "No tools available."

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_LANGGRAPH.format(tools_description=tools_description)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def agent_node_logic(state: AgentState):
        global _iteration_count_local
        _iteration_count_local += 1
        logger.info(f"--- AGENT NODE --- Iteration: {_iteration_count_local}/{AGENT_MAX_ITERATIONS}")

        if _iteration_count_local > AGENT_MAX_ITERATIONS:
            logger.warning(f"Max iterations ({AGENT_MAX_ITERATIONS}) reached. Returning AgentFinish.")
            return {
                "agent_outcome": AgentFinish(return_values={"output": "Reached maximum iterations. Please try rephrasing."}, log="Max iterations reached."),
                "chat_history": [AIMessage(content="Reached maximum iterations. Please try rephrasing.")],
                "current_tool_action": None, "current_tool_retries": 0, "last_tool_error": None,
                "retrieved_tool_context": None, "summarized_tool_context": None, "summarization_error": None,
            }

        final_context_for_prompt = None
        if state.get("summarized_tool_context"):
            final_context_for_prompt = state["summarized_tool_context"]
            logger.info("Agent node using SUMMARIZED tool context.")
        elif state.get("retrieved_tool_context"):
            final_context_for_prompt = state["retrieved_tool_context"]
            logger.info("Agent node using FULL retrieved tool context.")

        current_chat_history = list(state.get("chat_history", []))
        last_tool_error = state.get("last_tool_error")
        current_retries_for_failed_tool = state.get("current_tool_retries", 0)

        if last_tool_error and current_retries_for_failed_tool >= MAX_TOOL_RETRIES:
             tool_name_failed = state.get('current_tool_action').tool if state.get('current_tool_action') else 'UnknownTool'
             logger.info(f"Agent node informed of persistent tool failure for '{tool_name_failed}': {last_tool_error}")
             failure_context_message = AIMessage(
                 content=f"[Critical Tool Failure after {current_retries_for_failed_tool} retries for action '{tool_name_failed}']: {last_tool_error}\n"
                         f"You must now decide how to proceed. You can inform the user, try a different approach or tool, or ask for clarification."
             )
             current_chat_history.append(failure_context_message)

        if final_context_for_prompt:
            context_message = AIMessage(
                content=f"[Context from previous tool(s) to consider for next action or final response]:\n{final_context_for_prompt}"
            )
            extended_chat_history = current_chat_history + [context_message]
        else:
            extended_chat_history = current_chat_history

        llm_input_dict = {"input": state["input"], "chat_history": extended_chat_history}
        formatted_prompt_messages = agent_prompt.format_messages(**llm_input_dict)
        response_ai_message: AIMessage = llm_with_tools.invoke(formatted_prompt_messages)

        updates_for_state = {
            "retrieved_tool_context": None, "summarized_tool_context": None, "summarization_error": None,
            "chat_history": [response_ai_message]
        }

        if not response_ai_message.tool_calls:
            logger.info("Agent decided to FINISH.")
            updates_for_state["agent_outcome"] = AgentFinish(return_values={"output": response_ai_message.content}, log=str(response_ai_message))
            updates_for_state["current_tool_action"] = None
            updates_for_state["current_tool_retries"] = 0
            updates_for_state["last_tool_error"] = None
        else:
            actions = []
            first_tool_call = response_ai_message.tool_calls[0]
            current_action = AgentAction(
                tool=first_tool_call['name'], tool_input=first_tool_call['args'],
                log=f"Agent decision: call tool {first_tool_call['name']} with args {first_tool_call['args']}",
                tool_call_id=first_tool_call.get('id')
            )
            actions.append(current_action)
            logger.info(f"Agent decided to call tool: {current_action.tool}")
            updates_for_state["agent_outcome"] = actions
            updates_for_state["current_tool_action"] = current_action
            updates_for_state["current_tool_retries"] = 0
            updates_for_state["last_tool_error"] = None
        return updates_for_state

    tool_executor = ToolExecutor(tools_list)

    def tool_node_logic(state: AgentState):
        logger.info("--- TOOL NODE ---")

        current_tool_action_from_state = state.get("current_tool_action") # Action decided by agent, or action that previously failed.
        current_retries = state.get("current_tool_retries", 0) # Number of retries already attempted for current_tool_action_from_state.
        last_tool_error = state.get("last_tool_error") # Error from the *very last* tool execution attempt.

        action_to_execute: Optional[AgentAction] = None

        # Scenario 1: Retry a previously failed action.
        # This is identified if last_tool_error is set, current_tool_action_from_state is the action that failed,
        # and current_retries (which was incremented upon that failure) is > 0.
        # route_after_tools would have routed back to "tools" if retries < MAX_TOOL_RETRIES.
        if last_tool_error and current_tool_action_from_state and current_retries > 0:
            action_to_execute = current_tool_action_from_state
            # current_retries is the number of *previous* failed attempts.
            # So, this is retry attempt number `current_retries`.
            logger.info(f"Retrying tool: {action_to_execute.tool} (This is retry attempt {current_retries} of {MAX_TOOL_RETRIES})")

        # Scenario 2: Execute a new action selected by the agent.
        # This is identified if there's no last_tool_error (meaning the previous action, if any, was not an error in this node),
        # and current_tool_action_from_state is set (by agent_node_logic).
        # current_retries would be 0 if set by agent_node_logic for a new action.
        elif not last_tool_error and current_tool_action_from_state:
            action_to_execute = current_tool_action_from_state
            # Ensure current_retries is 0 for a fresh attempt, though agent_node should have set this.
            if current_retries != 0:
                logger.warning(f"Expected current_retries to be 0 for a new tool action, but found {current_retries}. Proceeding with action: {action_to_execute.tool}")
            logger.info(f"Attempting tool (selected by agent): {action_to_execute.tool} with input: {action_to_execute.tool_input}")

        # If action_to_execute is still None, it means there's a logic issue or unexpected state.
        if not action_to_execute:
            # This can happen if agent_outcome was AgentFinish but was routed here,
            # or if current_tool_action was not set by agent_node when it should have.
            agent_outcome_val = state.get('agent_outcome')
            error_msg = (f"Tool node: No valid action to execute. "
                         f"current_tool_action: {current_tool_action_from_state.tool if current_tool_action_from_state else 'None'}, "
                         f"last_tool_error: {'present' if last_tool_error else 'None'}, "
                         f"current_retries: {current_retries}, "
                         f"agent_outcome type: {type(agent_outcome_val).__name__}.")
            logger.error(error_msg + f" Full agent_outcome: {agent_outcome_val}")
            logger.error(error_msg) # Log the detailed error message
            # The original more generic error_msg for the user-facing message:
            user_facing_error_msg = "Tool node: No valid action to execute. This indicates a potential routing or state issue in the graph."
            return {
                "last_tool_error": user_facing_error_msg, # Keep it concise for state
                "retrieved_tool_context": None,
                "agent_outcome": AgentFinish(return_values={"output": f"System error: {user_facing_error_msg}. Cannot proceed."}, log=error_msg), # Log detailed error
                "chat_history": [AIMessage(content=f"A system error occurred: {user_facing_error_msg}. Unable to proceed with tool execution.")]
            }
        try:
            # On successful execution, current_tool_action and current_tool_retries from input state are no longer relevant
            # for the *next* agent step, so they are not explicitly cleared from state here.
            # The agent_node_logic will reset them if it decides on a *new* action.
            # If this tool call fails, they will be updated with new failure info.
            output = tool_executor.invoke(action_to_execute)
            logger.info(f"Tool {action_to_execute.tool} executed successfully. Output (truncated): {str(output)[:200]}")
            tool_message = ToolMessage(
                content=str(output), name=action_to_execute.tool,
                tool_call_id=getattr(action_to_execute, 'tool_call_id', None)
            )
            return {"chat_history": [tool_message], "last_tool_error": None, "retrieved_tool_context": str(output) }
        except Exception as e:
            error_message = f"Error executing tool '{action_to_execute.tool}': {e}"
            logger.exception(error_message)
            new_retry_count = state.get("current_tool_retries", 0) + 1
            error_tool_message = ToolMessage(
                content=f"Execution Error: {error_message}", name=action_to_execute.tool,
                tool_call_id=getattr(action_to_execute, 'tool_call_id', None)
            )
            return {"last_tool_error": error_message, "retrieved_tool_context": None,
                    "chat_history": [error_tool_message], "current_tool_retries": new_retry_count,
                    "current_tool_action": action_to_execute} # Preserve failing action for retry decision

    def call_summarizer_node_logic(state: AgentState, summarizer_graph_app):
        logger.info("--- CALL SUMMARIZER NODE ---")
        retrieved_context = state.get("retrieved_tool_context")
        original_user_query = state.get("input")
        if not retrieved_context or not retrieved_context.strip():
            logger.info("No retrieved_tool_context to summarize.")
            return {"summarized_tool_context": None, "summarization_error": "No context provided to summarize."}

        logger.info(f"Invoking summarizer graph for context of length {len(retrieved_context)}.")
        summarizer_input = {"context_string_to_summarize": retrieved_context, "original_query": original_user_query}
        summary_result_state = summarizer_graph_app.invoke(summarizer_input)
        summary = summary_result_state.get("summary")
        error = summary_result_state.get("error")

        if error:
            logger.error(f"Summarization error: {error}")
            return {"summarized_tool_context": None, "summarization_error": error, "retrieved_tool_context": retrieved_context}

        logger.info(f"Summarization successful. Summary length: {len(summary if summary else '')}")
        return {"summarized_tool_context": summary, "summarization_error": None, "retrieved_tool_context": retrieved_context}

    def route_after_agent(state: AgentState):
        logger.info(f"ROUTE_AFTER_AGENT: Iteration count from agent_node was {_iteration_count_local}")
        agent_outcome = state.get('agent_outcome')
        if isinstance(agent_outcome, AgentFinish):
            logger.info("Agent decided to FINISH. Routing to END.")
            return END
        elif isinstance(agent_outcome, list) and agent_outcome and all(isinstance(act, AgentAction) for act in agent_outcome):
             logger.info(f"Agent decided to call tool(s). Routing to 'tools'. Action: {agent_outcome[0].tool}")
             return "tools"
        else:
            logger.warning(f"Unknown or empty agent outcome in route_after_agent: {agent_outcome}. Routing to END.")
            return END

    def route_after_tools(state: AgentState):
        logger.info("ROUTE_AFTER_TOOLS: Evaluating tool execution outcome.")
        last_tool_error = state.get("last_tool_error")
        current_tool_action = state.get("current_tool_action") # This is the action that was just attempted
        current_retries = state.get("current_tool_retries", 0) # This was incremented by tool_node on failure

        if last_tool_error:
            tool_name = current_tool_action.tool if current_tool_action else "UnknownTool"
            logger.warning(f"Tool execution failed for '{tool_name}' on attempt {current_retries}. Error: {last_tool_error}")
            if current_retries < MAX_TOOL_RETRIES:
                logger.info(f"Routing for retry {current_retries + 1}/{MAX_TOOL_RETRIES} for tool '{tool_name}'.")
                # State for retry (current_tool_action, current_tool_retries) is already set by the failing tool_node.
                return "tools" # Retry the same tool action by re-entering the tool_node
            else:
                logger.error(f"Tool '{tool_name}' failed after {current_retries} attempts (max {MAX_TOOL_RETRIES} allowed). Routing to agent for error handling.")
                # Agent will see last_tool_error and current_tool_retries == MAX_TOOL_RETRIES.
                # agent_node_logic has specific handling to inform LLM about this persistent failure.
                return "agent"
        else:
            logger.info("Tool execution successful. Checking if summarization is needed.")
            retrieved_context = state.get("retrieved_tool_context", "")
            if isinstance(retrieved_context, str) and len(retrieved_context) > TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION:
                logger.info(f"Context length ({len(retrieved_context)}) > {TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION}. Routing to summarizer.")
                return "summarizer"
            else:
                logger.info(f"Context length ({len(retrieved_context)}) <= {TOOL_CONTEXT_MAX_CHARS_FOR_SUMMARIZATION}. Skipping summarizer, routing to agent.")
                return "agent"

    summarizer_llm = llm
    summarizer_graph_app = create_summarizer_graph(summarizer_llm)
    bound_call_summarizer_node = functools.partial(call_summarizer_node_logic, summarizer_graph_app=summarizer_graph_app)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node_logic)
    workflow.add_node("tools", tool_node_logic) # tool_executor is in closure
    workflow.add_node("summarizer", bound_call_summarizer_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_after_agent, { "tools": "tools", END: END })
    workflow.add_conditional_edges("tools", route_after_tools, {"tools": "tools", "summarizer": "summarizer", "agent": "agent"})
    workflow.add_edge("summarizer", "agent")

    app = workflow.compile()
    print("INFO: Main LangGraph agent graph (with retry and summarizer) compiled.")
    return app

if __name__ == '__main__':
    print("Testing graph.py structure (conceptual)...")
    # (Mock LLM and Tools for structural testing as before)
    print("graph.py structural test finished.")
