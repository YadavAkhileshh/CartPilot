import base64
import json
import os
import asyncpg
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

VECTOR_STORE = None

def get_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is None and PINECONE_INDEX_NAME:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_pinecone import PineconeVectorStore
            print("Connecting to Pinecone Vector Store...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            VECTOR_STORE = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        except Exception as e:
            print(f"Failed to load vector store: {e}")
    return VECTOR_STORE

@tool
async def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database using HYBRID SEARCH.
    It pre-filters by maximum price and/or organic status using PostgreSQL, 
    and then performs Semantic Vector Search (Pinecone) on the query.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    if not DATABASE_URL:
        return "[]"

    conn = await asyncpg.connect(DATABASE_URL)
    
    # Load preferences
    try:
        prefs_rows = await conn.fetch("SELECT key, value FROM user_preferences")
        prefs = {row["key"]: row["value"] for row in prefs_rows}
    except Exception:
        prefs = {}
    
    if is_organic is None and prefs.get("prefers_organic") == "True":
        is_organic = True
    if max_price is None and "max_price" in prefs:
        try:
            max_price = float(prefs["max_price"])
        except ValueError:
            pass

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params = []
    
    if max_price is not None:
        params.append(max_price)
        sql += f" AND price <= ${len(params)}"
        
    if is_organic is not None:
        params.append(1 if is_organic else 0)
        sql += f" AND is_organic = ${len(params)}"

    rows = await conn.fetch(sql, *params)
    await conn.close()

    sql_products = {
        row["id"]: {
            "id":          row["id"],
            "name":        row["name"],
            "category":    row["category"],
            "price":       row["price"],
            "description": row["description"],
            "is_organic":  bool(row["is_organic"]),
        }
        for row in rows
    }

    final_products = []
    if query:
        vector_store = get_vector_store()
        if vector_store:
            try:
                # Retrieve top 10 closest semantic matches asynchronously
                docs = await vector_store.asimilarity_search(query, k=10)
                
                for doc in docs:
                    p_id = int(doc.metadata.get("id", -1))
                    if p_id in sql_products:
                        final_products.append(sql_products[p_id])
                        
            except Exception as e:
                print(f"Vector search failed: {e}. Falling back to SQL ONLY.")
                like_query = query.lower()
                for p_id, p in sql_products.items():
                    if like_query in p["name"].lower() or like_query in p["description"].lower() or like_query in p["category"].lower():
                        final_products.append(p)
        else:
            like_query = query.lower()
            for p_id, p in sql_products.items():
                if like_query in p["name"].lower() or like_query in p["description"].lower() or like_query in p["category"].lower():
                    final_products.append(p)
    else:
        final_products = list(sql_products.values())

    return json.dumps(final_products)

@tool
async def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = await get_product_rating(product_id)
    return json.dumps(result)

@tool
async def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    if not DATABASE_URL:
        return "Error: Database not connected."
        
    conn = await asyncpg.connect(DATABASE_URL)
    row = await conn.fetchrow("SELECT name, price FROM products WHERE id = $1", product_id)

    if not row:
        await conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row["name"], row["price"]
    
    order_id = await conn.fetchval(
        "INSERT INTO orders (product_id, product_name, price) VALUES ($1, $2, $3) RETURNING id",
        product_id, name, price
    )
    await conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )

@tool
async def describe_product_image(image_path: str) -> str:
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

    response = await vision_llm.ainvoke([message])
    return response.content

@tool
async def get_order_history() -> str:
    """
    Retrieve the history of all orders placed by the user.
    Returns a JSON array of orders, each with: id, product_id, product_name, price, ordered_at.
    """
    if not DATABASE_URL:
        return "[]"
        
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch("SELECT id, product_id, product_name, price, ordered_at FROM orders")
    await conn.close()

    orders = [
        {
            "id":           row["id"],
            "product_id":   row["product_id"],
            "product_name": row["product_name"],
            "price":        row["price"],
            "ordered_at":   str(row["ordered_at"]),
        }
        for row in rows
    ]
    return json.dumps(orders)

@tool
async def save_user_preference(key: str, value: str) -> str:
    """
    Save or update a user preference to remember it across sessions.
    Supported keys:
    - 'prefers_organic': set to 'True' if the user always prefers organic products, or 'False' otherwise.
    - 'max_price': set to a numeric value (e.g., '20') if the user never wants items over that price limit.
    """
    if not DATABASE_URL:
        return "Error: Database not connected."
        
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute(
        """
        INSERT INTO user_preferences (key, value) VALUES ($1, $2)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        key.strip(), value.strip()
    )
    await conn.close()
    return f"Preference '{key}' has been saved as '{value}'."

async def get_user_preferences() -> dict:
    if not DATABASE_URL:
        return {}
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch("SELECT key, value FROM user_preferences")
        await conn.close()
        return {row["key"]: row["value"] for row in rows}
    except Exception:
        return {}

async def get_system_prompt() -> str:
    prefs = await get_user_preferences()
    
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
    
    base_prompt = (
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items. You MUST respect the active user preferences listed below.\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format:\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability.\n"
        "5. If only one product qualifies, ask: 'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy:\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product.\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"
        "USER PREFERENCES:\n"
        "1. Call save_user_preference to save long-term rules.\n"
        "2. Confirm to the user that you've saved their preference.\n\n"
        "Never place an order unless explicitly confirmed. Never guess a product_id.\n"
    )
    
    return base_prompt + prefs_text

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

async def check_guardrail(message: str) -> bool:
    if not message:
        return True
    
    cleaned = message.strip()
    if cleaned.startswith("I uploaded a product image"):
        return True
        
    prompt = f"""You are a guardrail classifier for a shopping assistant.
Classify as SHOPPING_RELATED or OFF_TOPIC. Respond ONLY with the classification.
User message: "{cleaned}"
Classification:"""
    
    try:
        response = await llm.ainvoke(prompt)
        res_text = response.content.strip().upper()
        if "OFF_TOPIC" in res_text and "SHOPPING_RELATED" not in res_text:
            return False
        if "OFF_TOPIC" in res_text and "SHOPPING_RELATED" in res_text:
            idx_off = res_text.rfind("OFF_TOPIC")
            idx_shop = res_text.rfind("SHOPPING_RELATED")
            if idx_off > idx_shop:
                return False
    except Exception:
        return True
    return True

class PersonalShopperAgent:
    async def ainvoke(self, input_data: dict, config=None):
        messages = input_data.get("messages", [])
        latest_user_msg = get_latest_user_message(messages)
        
        if latest_user_msg is not None:
            if not await check_guardrail(latest_user_msg):
                out_messages = list(messages)
                out_messages.append(AIMessage(content="I'm sorry, but I can only help you with shopping-related requests. How can I help you with your shopping today?"))
                return {"messages": out_messages}
        
        system_prompt = await get_system_prompt()
        
        compiled_agent = create_agent(
            tools=[search_products, get_rating, checkout, describe_product_image, get_order_history, save_user_preference],
            model=llm,
            system_prompt=system_prompt
        )
        
        return await compiled_agent.ainvoke(input_data, config)
    
    async def astream_events(self, input_data: dict, config=None, **kwargs):
        messages = input_data.get("messages", [])
        latest_user_msg = get_latest_user_message(messages)
        
        if latest_user_msg is not None:
            if not await check_guardrail(latest_user_msg):
                # Fake a stream event for guardrail block
                yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="I'm sorry, but I can only help you with shopping-related requests.")}}
                return
                
        system_prompt = await get_system_prompt()
        compiled_agent = create_agent(
            tools=[search_products, get_rating, checkout, describe_product_image, get_order_history, save_user_preference],
            model=llm,
            system_prompt=system_prompt
        )
        
        async for event in compiled_agent.astream_events(input_data, config, version="v1"):
            yield event

agent = PersonalShopperAgent()
