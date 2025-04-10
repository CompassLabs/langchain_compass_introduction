# Empower AI Agents with a Universal DeFi Toolkit: LangChain-Compass 1.0

In recent years, decentralized finance (DeFi) has surged in popularity, bringing traditional financial services such as lending, swapping, and yield farming to blockchain platforms. On the other hand, Large Language Models (LLMs) have rapidly advanced, offering powerful capabilities for understanding and generating human-like text.

**LangChain-Compass 1.0** is a new toolkit designed to bridge these two worlds. It provides LLM-based agents with a universal DeFi API, enabling them to securely and reliably interact with major DeFi protocols—including **Uniswap**, **Aave**, and **Aerodrome**—directly on-chain.

In this article, we'll walk through:

1. **What is LangChain_Compass 1.0 and why is matters?**
2. **How to use the toolkit to access DeFi tools.**
3. **Building a simple ReAct (Reasoning + Acting) agent using LangChain** that can execute on-chain actions.

Let's dive in.

# 1. What is LangChain-Compass 1.0? 


# import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
