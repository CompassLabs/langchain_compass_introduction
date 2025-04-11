from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.config import RunnableConfig
from rich.console import Console

from answer_types import AnswerType, ChatAnswer
import requests
import json
import uuid
from typing import Literal
from langchain_compass.toolkits import LangchainCompassToolkit

load_dotenv()


SUPPORTED_MODELS_TYPE = Literal["gpt-4o", "o1-2024-12-17", "gpt-4o-mini"]


def initialize_agent(
    model: str = "gpt-4o",
) -> CompiledGraph:
    """
    Initialize the agent with an LLM and a set of tools.

    Args:
    model: The model to use for the LLM. Should be one of the SUPPORTED_MODELS_TYPE.
    """
    llm = ChatOpenAI(model=model)

    # Get the tools. At the moment, this is just the Compass toolkit.
    tools = LangchainCompassToolkit(compass_api_key=None).get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()

    # Generate a unique id for this agent.
    agent_id: str = str(uuid.uuid4())

    # Create the agent with the LLM, tools, and memory.
    agent_executor = create_react_agent(
        llm,
        tools=tools,
        response_format=None,
        checkpointer=memory,
        prompt=(
            "You are a helpful agent that can interact onchain using APIs that you've been told. Most importantly the Compass API. "
            "Think step by step."
            "You will help users to make transactions on chain as well as return useful information and insights."
            "Always try to be concise. If you are missing information, always ask for it."
            "If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "If it's ambiguous which tool to use, always prefer Compass tools over Coinbase tools or Coingecko tools."
            "you must say so, and encourage them to implement it themselves using the Compass API. Then can contact Compass Labs at contact@compasslabs.ai."
        ),
    )
    agent_executor.id = agent_id  # type: ignore
    return agent_executor


def _get_trajectory(  # for eval.yaml # returns (response, trajectory)
    *, agent_executor: CompiledGraph, user_input: str, thread_id: str
):
    config = RunnableConfig(configurable={"thread_id": thread_id})
    r = agent_executor.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=config
    )
    return r["messages"][-1].content, [
        m.name for m in r["messages"] if isinstance(m, ToolMessage)
    ]


def _determine_answer_type(messages: list[BaseMessage]) -> AnswerType:
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage):
        if last_message.status == "error":
            return AnswerType.ERROR
        if last_message.status == "success":
            data = json.loads(last_message.model_dump_json())
            t = (
                AnswerType.IMAGE
                if data["type"] == "image"
                else AnswerType.UNSIGNED_TRANSACTION
            )
            return t
        raise ValueError(f"Unknown status: {last_message.status}")
    return AnswerType.TEXT


def _get_non_stream_response_with_trajectory(
    *, agent_executor: CompiledGraph, user_input: str, thread_id: str
) -> list[ChatAnswer]:
    """
    Retrieve a non-streaming response with trajectory information from the agent.

    Args:
        agent_executor: The compiled graph of the agent.
        user_input: The input message from the user.
        thread_id: The unique identifier for the conversation thread.

    Returns:
        A list of ChatAnswer objects containing the response.
    """
    # Configure the RunnableConfig with the thread_id
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # Invoke the agent with the user input
    r = agent_executor.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=config
    )

    # Determine the type of answer returned by the agent
    answer_type: AnswerType = _determine_answer_type(messages=r["messages"])

    # Handle IMAGE or UNSIGNED_TRANSACTION answer types
    if answer_type in [AnswerType.IMAGE, AnswerType.UNSIGNED_TRANSACTION]:
        data = json.loads(r["messages"][-1].content)

        # Create a completion message for the agent
        completion = f"""
                Assume you give a correct answer to the following prompt: {user_input}.
                Phrase a short message to put on top of the answer. Something like 'Here is...'
        """
        # Invoke the agent with the completion message
        rr = agent_executor.invoke(
            {"messages": [HumanMessage(content=completion)]}, config=config
        )
        content = (
            data["content"]
            if answer_type == AnswerType.UNSIGNED_TRANSACTION
            else data["image"]
        )
        return [
            ChatAnswer(type=AnswerType.TEXT, content=rr["messages"][-1].content),
            ChatAnswer(type=answer_type, content=content),
        ]
    if answer_type == AnswerType.ERROR:
        return [ChatAnswer(type=AnswerType.TEXT, content=r["messages"][-1].content)]
    if answer_type == AnswerType.TEXT:
        return [ChatAnswer(type=AnswerType.TEXT, content=r["messages"][-1].content)]
    else:
        raise ValueError(f"Unknown answer type: {answer_type}")


def get_non_stream_response(
    *, agent_executor: CompiledGraph, user_input: str, thread_id: str
) -> list[ChatAnswer]:
    """
    Retrieve a non-streaming response from the agent.

    Args:
        agent_executor: The compiled graph of the agent.
        user_input: The input message from the user.
        thread_id: The unique identifier for the conversation thread.

    Returns:
        A list of ChatAnswer objects containing the response.
    """

    try:
        return _get_non_stream_response_with_trajectory(
            agent_executor=agent_executor, user_input=user_input, thread_id=thread_id
        )
    except Exception as e:  # noqa: E722
        # TODO: this should raise an all-quite alert, with the conversation thread and user-input.
        # Send an alert to All Quiet
        url = "https://allquiet.app/api/webhook/fcbd4705-ccee-44b5-bd65-1e886014541a"

        # Create a payload to send to All Quiet
        payload = json.dumps(
            {
                "Status": "Open",
                "Severity": "Critical",
                "Title": "Unexpected ERROR in AI answer.",
                "agent_id": agent_executor.id,  # type: ignore
                "thread_id": thread_id,  # type: ignore
                "user_input": user_input,
                "url": "https://smith.langchain.com",
                "error_message": str(e),
            }
        )
        headers = {"Content-Type": "application/json"}

        # Send the request to All Quiet
        response = requests.request("POST", url, headers=headers, data=payload)

        # Print the response
        print(response.text)

        # Return a default error message
        return [
            ChatAnswer(
                type=AnswerType.TEXT,
                content="Something went wrong on Compass AI side. Please mail contact@compasslabs.ai to report this issue.",
            )
        ]


if __name__ == "__main__":
    test_messages = [
        "Please set my DAI allowance to 5.5 DAI for the UniswapV3Router for my wallet 0x7Fd9DBad4d8B8F97BEdAC3662A0129a5774AdA8E on arbitrum.",
        "What is my current USDC allowance on AAVE on ethereum? My address is 0x7Fd9DBad4d8B8F97BEdAC3662A0129a5774AdA8E.",
        "Can you visualize my portfolio on arbitrum? My address is 0x7Fd9DBad4d8B8F97BEdAC3662A0129a5774AdA8E.",
        "Please use the aave_supply_ tool on Arbitrum Mainnet with my wallet address, 0x7Fd9DBad4d8B8F97BEdAC3662A0129a5774AdA8E, as both the sender and on-behalf-of address, supply 0.1 USDT, and ensure all transactions are performed using the correct chain and asset details.",
        "can you visualize the portfolio of 0x7Fd9DBad4d8B8F97BEdAC3662A0129a5774AdA8E on Base",
    ]

    agent_executor = initialize_agent()
    responses: list[ChatAnswer] = get_non_stream_response(
        agent_executor=agent_executor,
        user_input=test_messages[5],
        thread_id=str(uuid.uuid4()),
    )
    console = Console()
    for answer in responses:
        console.print(answer)