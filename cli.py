import sys

from dotenv import load_dotenv
from langgraph.graph.graph import CompiledGraph
from agent import initialize_agent, get_non_stream_response
import hashlib
import time
from answer_types import ChatAnswer
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()


# Chat Mode
def run_chat_mode(*, agent_executor: CompiledGraph, thread_id: str) -> None:
    """
    Run the agent interactively based on user input.

    This method starts an interactive loop where the user can enter prompts
    and the agent will respond. The loop can be exited by entering the
    command 'exit'.
    """
    console = Console()
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Get the response from the agent
            answers: list[ChatAnswer] = get_non_stream_response(
                agent_executor=agent_executor,
                user_input=user_input,
                thread_id=thread_id,
            )

            # Print the response
            for answer in answers:
                console.print(Markdown(str(answer.content)))  # pyright: ignore

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


if __name__ == "__main__":
    thread_id: str = hashlib.sha256(str(time.time()).encode()).hexdigest()
    agent_executor = initialize_agent("gpt-4o-mini")

    run_chat_mode(agent_executor=agent_executor, thread_id=thread_id)
