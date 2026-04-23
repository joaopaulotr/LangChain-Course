from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core import ChatMessage, SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable
load_dotenv()

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"

# ---- Tools ---------------------------

@tool
def get_product_price(product:str)-> float:
    """Get the price of a product on the catalog."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99
    }
    return prices.get(product, 0)

@tool 
def apply_discount(price:float, discount_tier:str)-> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# ---- Tools ---------------------------

# ---- Agent Loop ---------------------------

@traceable(name="LangChain Agent Loop")

# Query================================
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage
                (
                    content= """ 
                 You are a helpful assistant that answers questions about product prices and discounts.
                 You have acess to a product catalog tool and a discount tool. Always use the tools to get accurate information about prices and discounts.
                 STRIC RULES - You must follow exactly 
                 Never make up prices or discounts, always use the tools to get accurate information."""
                 )
                ,
        HumanMessage(content=question)
                ]
# Query================================

# Agent Loop================================
    for iteration in range(1, MAX_ITERATIONS):
        print(f"Iteration {iteration}")

        # Action================================
        ai_message = llm_with_tools.invoke(messages)
        # Action================================
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"Final annswer: {ai_message.content}")
            # Answer================================
            return ai_message.content
            # Answer================================

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool selectecd: {tool_call} with args {tool_args}, call id: {tool_call_id}")
# Tool Call================================
        tool_to_use = tools_dict.get(tool_name)
# Tool Call================================   
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
 
# Observation================================   
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool result: {observation}")
# Observation================================   
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer.")
    return None
# Agent Loop================================


if __name__ == "__main__":
    print("Welcome to the Product Price Agent!")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
    print()
    print(f"Final answer: {result}")