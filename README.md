# Empower AI Agents with a Universal DeFi Toolkit: LangChain-Compass 1.0

In recent years, decentralized finance (DeFi) has surged in popularity, bringing traditional financial services such as lending, swapping, and yield farming to blockchain platforms. On the other hand, Large Language Models (LLMs) have rapidly advanced, offering powerful capabilities for understanding and generating human-like text.

**LangChain-Compass 1.0** is a new toolkit designed to bridge these two worlds. It provides LLM-based agents with a universal DeFi API, enabling them to securely and reliably interact with major DeFi protocols—including **Uniswap**, **Aave**, and **Aerodrome**—directly on-chain.

In this article, we'll walk through:

1. **What is LangChain_Compass 1.0 and why is matters?**
2. **How to use the toolkit to access DeFi tools.**
3. **Building a simple ReAct (Reasoning + Acting) agent using LangChain** that can execute on-chain actions.

Let's dive in.

# 1. What is LangChain-Compass 1.0? 
The **LangChain-Compass** toolkit gives LLM's access to a [Universal DeFi API](https://docs.compasslabs.ai/documentation/getting-started). This means LLM-based agents can perform actions like:

- Swap tokens on major DEXs like Uniswap
- Lend or borrow assets using protocols such as Aave
- Provide liquidity on Aerodrome and Uniswap
- Transfer funds between wallets.
- Query balances, portfolios and positions

To see an LLM-Application built using the toolkit please checkout the [Compass AI here](https://gpt.compasslabs.ai/)

# 2. How to Use the Toolkit

To begin, install and import the toolkit in your Python environment.
```bash
pip install -qU langchain-compass
```

Then, you can list the available tools provided by LangChain-Compass simply by running:

```python
from langchain_compass.toolkits import LangchainCompassToolkit
tools = LangchainCompassToolkit(compass_api_key=None).get_tools()
[t.name for t in tools]
```

# 3. Tools Overview:

Here is a table presenting the different tools:

[TABLE HERE]


# 4. Building a Simple ReAct Agent Using LangChain

LangChain is a popular framework for building AI agents that can process and respond to language inputs. The ReAct paradigm—short for “Reason and Act”—encourages the agent to think step-by-step before taking an action. Here’s how you can set up a LangChain ReAct agent equipped with the LangChain-Compass DeFi tools.

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_compass.toolkits import LangchainCompassToolkit
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables (optional, for API keys, etc.)
load_dotenv()

# Initialize LLM - replace 'gpt-4o' with a model of your choice
llm = ChatOpenAI(model='gpt-4o')

# Get the DeFi tools from LangchainCompassToolkit
tools = LangchainCompassToolkit(compass_api_key=None).get_tools()

# Setup memory for your agent
memory = MemorySaver()

# Create a ReAct agent with the specified LLM, tools, and memory
agent = create_react_agent(
    llm,
    tools=tools,
    checkpointer=memory,
    prompt="You are a helpful agent that can interact onchain using tools that you've been told how to use."
)

# Example user query
from langchain_core.messages import HumanMessage
user_input = 'what is the balance of vitalic.eth.'

# Optional config data, such as thread IDs or session context
config = {"configurable": {"thread_id": "abc123"}}

# Invoke the agent with the user query
output = agent.invoke(input={"messages": [HumanMessage(content=user_input)]}, config=config)

# Display the agent's final response
print(output["messages"][-1].content)
```






# import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
