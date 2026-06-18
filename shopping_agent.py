import base64
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database using HYBRID SEARCH.
    It pre-filters by maximum price and/or organic status using SQLite, 
    and then performs Semantic Vector Search (FAISS) on the query.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    # Load user preferences and apply as defaults if not explicitly provided
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
    if cursor.fetchone():
        cursor.execute("SELECT key, value FROM user_preferences")
        prefs = {row[0]: row[1] for row in cursor.fetchall()}
        if is_organic is None and prefs.get("prefers_organic") == "True":
            is_organic = True
        if max_price is None and "max_price" in prefs:
            try:
                max_price = float(prefs["max_price"])
            except ValueError:
                pass

    # 1. SQL Pre-filtering (Hard constraints)
    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    # Create a dictionary of SQL-allowed products for fast lookup
    sql_products = {
        row[0]: {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    }

    # 2. Semantic Vector Search (FAISS)
    final_products = []
    if query:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            
            # Retrieve top 10 closest semantic matches
            docs = vector_store.similarity_search(query, k=10)
            
            # Intersect: Only keep FAISS results that passed the SQL pre-filter
            for doc in docs:
                p_id = doc.metadata.get("id")
                if p_id in sql_products:
                    final_products.append(sql_products[p_id])
                    
        except Exception as e:
            print(f"Vector search failed: {e}. Falling back to SQL ONLY.")
            # Fallback to simple python filtering if FAISS isn't ready
            like_query = query.lower()
            for p_id, p in sql_products.items():
                if like_query in p["name"].lower() or like_query in p["description"].lower() or like_query in p["category"].lower():
                    final_products.append(p)
    else:
        # If no semantic query, just return the SQL filtered list
        final_products = list(sql_products.values())

    return json.dumps(final_products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Order History & Preferences Tools & Helpers
# ---------------------------------------------------------------------------

@tool
def get_order_history() -> str:
    """
    Retrieve the history of all orders placed by the user.
    Returns a JSON array of orders, each with: id, product_id, product_name, price, ordered_at.
    Use this when the user asks "what have I ordered before?" or about their previous orders.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, product_id, product_name, price, ordered_at FROM orders")
    rows = cursor.fetchall()
    conn.close()

    orders = [
        {
            "id":           row[0],
            "product_id":   row[1],
            "product_name": row[2],
            "price":        row[3],
            "ordered_at":   row[4],
        }
        for row in rows
    ]
    return json.dumps(orders)


@tool
def save_user_preference(key: str, value: str) -> str:
    """
    Save or update a user preference to remember it across sessions.
    Supported keys:
    - 'prefers_organic': set to 'True' if the user always prefers organic products, or 'False' otherwise.
    - 'max_price': set to a numeric value (e.g., '20') if the user never wants items over that price limit.
    Use this when the user explicitly expresses a long-term preference (e.g. 'I only buy organic', 
    'never show me items over $20', 'remember that I prefer organic').
    Returns a confirmation message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Ensure table exists
    cursor.execute("CREATE TABLE IF NOT EXISTS user_preferences (key TEXT PRIMARY KEY, value TEXT)")
    cursor.execute(
        "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
        (key.strip(), value.strip()),
    )
    conn.commit()
    conn.close()
    return f"Preference '{key}' has been saved as '{value}'."


def get_user_preferences() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
    if not cursor.fetchone():
        cursor.execute("CREATE TABLE user_preferences (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
    cursor.execute("SELECT key, value FROM user_preferences")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_system_prompt() -> str:
    # 1. Load preferences
    prefs = get_user_preferences()
    
    # 2. Build preferences text block
    prefs_lines = []
    if prefs:
        for k, v in prefs.items():
            if k == "prefers_organic" and v.lower() == "true":
                prefs_lines.append("- The user prefers organic products.")
            elif k == "max_price":
                prefs_lines.append(f"- The user never wants items over ${v}.")
            else:
                prefs_lines.append(f"- {k}: {v}")
    
    prefs_text = ""
    if prefs_lines:
        prefs_text = "\nACTIVE USER PREFERENCES (apply these unless the user explicitly overrides them in their query):\n" + "\n".join(prefs_lines) + "\n"
    
    # 3. Base system prompt
    base_prompt = (
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items. You MUST respect the active user preferences listed below. "
        "   Apply any price/organic preferences as default arguments to search_products (e.g. if user prefers organic, set is_organic=True; "
        "   if they have a max price constraint, set max_price to that limit), unless the user explicitly overrides them in their current message.\n"
        "   Note that search_products uses keyword matching on descriptions, so it may return some irrelevant items (e.g., searching 'honey' might return granola containing honey). "
        "   You MUST filter out any products that do not match the user's intended category or product type (e.g., if they want honey, do not list granola) before presenting them.\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"
        "USER PREFERENCES — when the user expresses a long-term preference (e.g., 'I only buy organic', "
        "'never show me items over $20', 'always prefer organic'):\n"
        "1. Call save_user_preference to save it.\n"
        "2. Confirm to the user that you've saved their preference and will remember it for future sessions.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message.\n"
    )
    
    return base_prompt + prefs_text


# ---------------------------------------------------------------------------
# Guardrails & Agent Wrapper
# ---------------------------------------------------------------------------

def get_latest_user_message(messages) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content")
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif hasattr(msg, "role") and msg.role == "user":
            return msg.content
        elif isinstance(msg, HumanMessage):
            return msg.content
    return None


def check_guardrail(message: str) -> bool:
    """
    Returns True if the message is shopping-related, False otherwise.
    """
    if not message:
        return True
    
    cleaned = message.strip()
    if cleaned.startswith("I uploaded a product image"):
        return True
        
    prompt = f"""You are a guardrail classifier for a shopping assistant.
Determine if the user's message is related to shopping, products, orders, preferences, or interacting with the shopping assistant (including greetings like 'hi' or 'hello' and asking what the assistant can do).

Classify as SHOPPING_RELATED if the message is:
- Searching for products (e.g. "I want organic honey")
- Inquiring about previous orders (e.g. "what did I buy before?")
- Stating preferences (e.g. "I only buy organic", "max price is $20")
- Placing an order or checking out (e.g. "yes", "please order it", "order number 2")
- Requesting ratings or product info (e.g. "tell me about product 5")
- Greeting or polite conversation (e.g. "hi", "hello", "thanks")
- Asking what the assistant does

Classify as OFF_TOPIC if the message is:
- Asking for general knowledge/facts (e.g. "what is the capital of France?", "who is Einstein?")
- Asking for weather (e.g. "what's the weather?")
- Asking to write poems, stories, essays, code, etc.
- Solving math/logic puzzles unrelated to the shopping products
- Asking general programming questions

Respond with ONLY the classification: either SHOPPING_RELATED or OFF_TOPIC. Do not write any other words.

User message: "{cleaned}"
Classification:"""
    
    try:
        response = llm.invoke(prompt)
        res_text = response.content.strip()
        # Remove think block if present
        if "<think>" in res_text.lower():
            parts = res_text.lower().split("</think>")
            if len(parts) > 1:
                res_text = parts[-1].strip()
        res_text = res_text.upper()
        if "OFF_TOPIC" in res_text and "SHOPPING_RELATED" not in res_text:
            return False
        if "OFF_TOPIC" in res_text and "SHOPPING_RELATED" in res_text:
            idx_off = res_text.rfind("OFF_TOPIC")
            idx_shop = res_text.rfind("SHOPPING_RELATED")
            if idx_off > idx_shop:
                return False
    except Exception as e:
        print(f"Guardrail error: {e}")
        # Default to True on failure
        return True
    return True


class PersonalShopperAgent:
    def invoke(self, input_data: dict, config=None):
        # 1. Input Guardrail check
        messages = input_data.get("messages", [])
        latest_user_msg = get_latest_user_message(messages)
        
        if latest_user_msg is not None:
            if not check_guardrail(latest_user_msg):
                out_messages = list(messages)
                redirect_content = (
                    "I'm sorry, but I can only help you with shopping-related requests, "
                    "such as searching for products, viewing ratings, managing your order history, "
                    "or placing orders. How can I help you with your shopping today?"
                )
                out_messages.append(AIMessage(content=redirect_content))
                return {"messages": out_messages}
        
        # 2. Get preferences & build dynamic system prompt
        system_prompt = get_system_prompt()
        
        # 3. Create the LangGraph agent
        compiled_agent = create_agent(
            tools=[search_products, get_rating, checkout, describe_product_image, get_order_history, save_user_preference],
            model=llm,
            system_prompt=system_prompt
        )
        
        # 4. Invoke the agent
        return compiled_agent.invoke(input_data, config)


agent = PersonalShopperAgent()

if __name__ == "__main__":

    # image_path = os.path.join("resources","oats.png")
    # response = describe_product_image(image_path)
    # print(response)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
