# Empower AI Agents with a Universal DeFi Toolkit: LangChain-Compass 1.0

In this tutorial we will build an agent that can execute transactions on Uniswap when given instructions from a user.

## 1. Introduction

Decentralized Financial platforms like Uniswap, Aave, and Aerodrome enable users to lend or borrow funds, speculate on asset price movements using derivatives, trade cryptocurrencies, and much much more.

LLM's don't need any introduction.

LangChain-Compass is a new toolkit designed to bridge these two worlds.

## 2. What is LangChain Compass

The LangChain-Compass toolkit gives LLM's access to a Universal DeFi API. This means LLM-based agents can perform actions like:

- Swap tokens on major DEXs like Uniswap

- Lend or borrow assets using protocols such as Aave

- Provide liquidity on Aerodrome and Uniswap

- Transfer funds between wallets.

- Query balances, portfolios and positions

To see a full stack LLM-Application built using the toolkit please checkout the [Compass AI here](https://gpt.compasslabs.ai/)

## 2. How to Use the Toolkit

To begin, import the toolkit in your Python environment.

```bash
pip install -qU langchain-compass
```


Then, you can list the available tools provided by LangChain-Compass simply by running:

```python
from langchain_compass.toolkits import LangchainCompassToolkit
tools = LangchainCompassToolkit(compass_api_key=None).get_tools()
[t.name for t in tools]
```
```bash
# output
aave_supply_
aave_borrow_
aave_repay_
aave_withdraw_
aave_asset_price_get_
aave_liquidity_change_get_
aave_user_position_summary_get_
aave_user_position_per_token_get_
aerodrome_slipstream_swap_sell_exactly_
aerodrome_slipstream_swap_buy_exactly_
aerodrome_slipstream_liquidity_provision_mint_
aerodrome_slipstream_liquidity_provision_increase_
aerodrome_slipstream_liquidity_provision_withdraw_
aerodrome_slipstream_liquidity_provision_positions_get_
...
```

Each tool corresponds to an [endpoint of the Compass API](https://docs.compasslabs.ai/api-reference/endpoints/aave-v3/supplylend) (the Universal API for DeFi).

## 3. Tools Overview:

Here is a table presenting the different tools:

[View PDF](./tool_table.pdf)
[View SVG](./tool_table.svg)

## 4. Building a Simple ReAct Agent Using LangChain

LangChain is a popular framework for building AI agents that can process and respond to language inputs. The ReAct paradigm—short for “Reason and Act”—encourages the agent to think step-by-step before taking an action. 

Here’s how you can set up a LangChain ReAct agent equipped with the LangChain-Compass DeFi tools:

1. To get started, you’ll need to add your OpenAI API key to a .env file in the root of your project. This keeps your credentials secure and separate from your codebase.

```
OPENAI_API_KEY=your_openai_api_key_here
```

You can use our [.env.example file](https://github.com/CompassLabs/langchain_compass_introduction/.env.example) as a template. Just copy it, rename it to .env, and replace the placeholder with your actual API key:

2. Run this script:
```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_compass.toolkits import LangchainCompassToolkit
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
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


## 5. Run the agent interactively based on user input.

To run the agent interactively please add this snippet to the bottom of the code in the previous section.

```python
from rich.console import Console
from rich.markdown import Markdown
console = Console()
print("Starting chat mode... Type 'exit' to end.")
while True:
    user_input = input("\nPrompt: ")
    output = agent.invoke(input = {"messages": [HumanMessage(content=user_input)]}, config=config)
    answer = output["messages"][-1].content
    console.print(Markdown(answer))
```

## 6. Full Implementation

To see a full implementation of the agent using the langchain-compass toolkit please check out [our github repo here](https://github.com/CompassLabs/langchain_compass_introduction/):

To install it you can run:

```
git clone https://github.com/CompassLabs/langchain_compass_introduction
cd langchain_compass_introduction
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements
```

And run:

```python ./cli.py```

This full implementation contains clauses for handling different answer types including unsigned transactions, pure text answers and more.

## 7. Conclusion.

`LangChain_Compass` offers a unified DeFi toolkit that empowers AI agents to seamlessly interact with platforms like Uniswap, Aave, and more. By integrating these tools into a ReAct agent built on LangChain, developers can create applications capable of executing on-chain transactions, querying balances, and providing liquidity. This approach simplifies the process of building advanced, interactive DeFi applications driven by large language models