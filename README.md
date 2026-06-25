# CartPilot: AI-Powered Shopping Agent

**A production-grade, cloud-native conversational shopping assistant built with FastAPI, LangChain, and a fully managed cloud data layer.**

---

## Why I Built This

When I started learning AI engineering, most tutorials show you how to build agents that run on your laptop with local SQLite databases and in-memory vector stores. That is great for learning the concepts, but I kept asking myself a question: what does this actually look like when it runs in production?

Every tutorial I followed produced something that would break the moment I deployed it. The SQLite file disappears when the server restarts. The FAISS index is gone on every cold start. Session memory is stored in a Python dictionary so every restart wipes out all conversations. These are not real applications. They are demos.

I built CartPilot to answer that question for myself. I wanted to go from prototype to something that could be deployed on a real cloud platform, survive server restarts, handle multiple users at the same time, and not lose any data the moment the process dies. This project is my attempt at building the real thing, not just the tutorial version.

---

## What It Does

CartPilot is a conversational shopping assistant. A user can describe what they want in plain English and the AI will search a product catalog, look up ratings, and place orders. The entire reasoning loop is handled by a LangChain ReAct agent. The agent decides which tools to call, in what order, and how to combine the results.

A user can also upload a photo of a product. The agent will use a vision model to identify what is in the photo and automatically search the store for similar items.

The agent remembers user preferences across sessions. If a user says they only want organic products or their maximum budget is $10, the agent writes that to the database and applies it automatically in every future conversation.

---

## Architecture

The flow of a single request through the system:

```mermaid
flowchart TD
    A["User sends message in browser"] --> B["FastAPI POST /api/chat"]
    B --> C["Upstash Redis: fetch session history"]
    C --> D["Full conversation history assembled"]
    D --> E{"Input Guardrail: LLM classification"}
    E -->|"OFF_TOPIC"| F["Reject with redirect message, stream back to user"]
    E -->|"SHOPPING_RELATED"| G["PersonalShopperAgent.ainvoke"]
    G --> H["get_system_prompt: load user preferences from Supabase"]
    H --> I["LangChain create_agent: Llama 3.3 70B with ReAct loop"]
    I --> J{"Agent reasons: which tool is needed?"}
    J --> K["search_products"]
    J --> L["get_rating"]
    J --> M["checkout"]
    J --> N["get_order_history"]
    J --> O["save_user_preference"]
    J --> P["describe_product_image"]
    K --> Q["Stage 1: asyncpg SQL filter in Supabase PostgreSQL"]
    Q --> R["Stage 2: Pinecone asimilarity_search semantic match"]
    R --> S["Intersect both results: hybrid ranking output"]
    L --> T["asyncpg AVG rating query from reviews table"]
    M --> U["asyncpg INSERT INTO orders RETURNING id"]
    N --> V["asyncpg SELECT all from orders table"]
    O --> W["asyncpg UPSERT into user_preferences table"]
    P --> X["Base64 encode image, call Llama 4 Scout vision model"]
    S --> Y["Agent observes results, reasons, loops or finalizes"]
    T --> Y
    U --> Y
    V --> Y
    W --> Y
    X --> Y
    Y --> Z["Final answer assembled by agent"]
    Z --> AA["Server streams response word by word over SSE"]
    AA --> AB["Upstash Redis: save updated history"]
    AB --> AC["Browser renders typing animation"]
```

---

## How the ReAct Agent Works

ReAct stands for Reasoning and Acting. It is a prompting pattern where the language model does not just respond to a message directly. Instead it goes through a structured loop until it has enough information to give a final answer.

The loop looks like this:

```
Thought  -->  the model reasons about what it needs to do next
Action   -->  the model picks a tool and calls it with specific arguments
Observation  -->  the model reads the tool output
Thought  -->  reasons about the observation and decides what to do next
... repeats ...
Final Answer  -->  the model has enough information, returns the response
```

Here is a real example of what happens inside the agent when a user types "I want organic honey under $20":

```
Thought: The user wants organic honey with a price constraint.
         I should search for products matching these criteria.

Action: search_products(query="honey", max_price=20.0, is_organic=True)

Observation: [{"id": 1, "name": "Organic Raw Honey", "price": 14.99, ...},
              {"id": 5, "name": "Organic Buckwheat Honey", "price": 18.99, ...},
              {"id": 7, "name": "Organic Acacia Honey", "price": 17.99, ...}]

Thought: I have 3 products. I need to get ratings for each to present to the user.

Action: get_rating(product_id=1)
Observation: {"average_rating": 4.62, "review_count": 4}

Action: get_rating(product_id=5)
Observation: {"average_rating": 4.4, "review_count": 4}

Action: get_rating(product_id=7)
Observation: {"average_rating": 4.5, "review_count": 4}

Thought: I now have products and ratings. I can format the final response.

Final Answer:
#1. Organic Raw Honey (ID:1) - $14.99 star 4.62 - organic
#2. Organic Acacia Honey (ID:7) - $17.99 star 4.5 - organic
#3. Organic Buckwheat Honey (ID:5) - $18.99 star 4.4 - organic
```

That entire loop, including the decision to call get_rating multiple times, happens autonomously. I did not write any code that says "after searching, look up ratings". The model figures that out by itself based on the system prompt instructions.

LangChain's create_agent function implements this ReAct loop. I built a custom wrapper class called PersonalShopperAgent on top of it so I could run the guardrail check and load user preferences before handing off to the agent.

---

## Technology Stack

### Backend Runtime

FastAPI with Uvicorn as the ASGI server. FastAPI is natively asynchronous so it can handle many concurrent requests without blocking. Every database call and every model call in this project uses Python async/await. Nothing blocks the event loop.

### LangChain Tools

The agent has six tools available. Each tool is an async Python function decorated with @tool from LangChain.

**search_products** - Takes a query string, optional max_price, and optional is_organic flag. Runs a two-stage hybrid search: first SQL in PostgreSQL for hard filtering, then Pinecone vector search for semantic matching. Returns a JSON array of matching products.

**get_rating** - Takes a product ID. Queries the reviews table in PostgreSQL to calculate the average rating and total review count for that product.

**checkout** - Takes a product ID. Looks up the product in PostgreSQL, inserts a row into the orders table, and returns an order confirmation with the generated order ID.

**get_order_history** - No arguments. Fetches all rows from the orders table and returns them as a JSON array. The agent calls this when the user asks about previous purchases.

**save_user_preference** - Takes a key and value. Upserts a row in the user_preferences table in PostgreSQL. Supported keys are prefers_organic and max_price. These preferences are loaded into the system prompt on every request so the agent applies them automatically.

**describe_product_image** - Takes an image file path. Reads the file, encodes it as Base64, and sends it to the Llama 4 Scout vision model on Groq. Returns a JSON object with the identified product type, a search query string, and whether it appears to be organic.

### Language Models

The reasoning and text model is Llama 3.3 70B served through Groq. I chose Groq because of its inference speed which is significantly faster than most other providers for Llama-family models.

The vision model is Llama 4 Scout served through Groq. It accepts a Base64-encoded image alongside a text prompt in the same API call and returns structured product information.

### Hybrid Search

Product search uses two stages working together.

Stage one is SQL filtering in Supabase PostgreSQL. The query builds a WHERE clause dynamically based on the price limit and organic preference. This is a hard filter, only products that pass these constraints are considered for the next stage.

Stage two is semantic vector search in Pinecone. The user query is embedded using the all-MiniLM-L6-v2 sentence transformer model which produces a 384-dimensional vector. Pinecone returns the top 10 products whose vectors are closest to the query vector using cosine similarity.

The final result is the intersection of both stages. A product must appear in the SQL results and the Pinecone results to be returned. This is more accurate than either method alone because SQL cannot understand meaning and vector search cannot enforce a price constraint.

### Embeddings

The all-MiniLM-L6-v2 model from sentence-transformers creates the embeddings. I chose this model because it is small enough to run locally without a GPU, produces high-quality semantic embeddings for short text, and outputs 384-dimensional vectors which are efficient to store and query in Pinecone.

Each product is embedded as a single text string combining its name, category, description, organic status, and price. This gives the vector enough context to match a natural language query like "something sweet and natural" to the honey category.

### Relational Database

Supabase provides a managed PostgreSQL instance. The schema has four tables.

The products table stores the catalog with fields for id, name, category, price, description, and is_organic.

The reviews table stores customer reviews linked to products with a foreign key. The get_rating tool queries this table with an aggregate AVG function.

The orders table records every checkout event with the product ID, name, price, and a timestamp that defaults to the current time at insert.

The user_preferences table is a simple key-value store with the key as the primary key so that saving a preference is a single UPSERT operation.

I use asyncpg to connect to PostgreSQL. asyncpg is a pure async PostgreSQL driver which means it never makes a blocking call. This is important because a single blocking database call would freeze the entire event loop and prevent other requests from being processed while waiting.

### Vector Database

Pinecone stores the product embeddings. I created a serverless index with cosine similarity and 384 dimensions to match the MiniLM embedding output size. The setup_db.py script reads all products from PostgreSQL, generates embeddings, and pushes them to Pinecone. At query time the langchain_pinecone PineconeVectorStore wrapper handles embedding the query and calling Pinecone's API.

### Session Memory and Stateless Design

Upstash Redis stores the full conversation history for each user session as a JSON-serialized list. The key is "session:" followed by the session ID.

The session ID is generated once in the browser using crypto.randomUUID() and stored in localStorage. This means the same user gets the same session ID across page refreshes and browser tabs on the same device.

Because the conversation history is in Redis and not in the server process memory, the application is completely stateless. If the server restarts, crashes, or scales to multiple instances, the next request from any user will still have access to their full conversation history by fetching it from Redis at the start of every request.

### Input Guardrails

Every incoming message goes through a classification call before the ReAct agent runs. The same Llama 3.3 70B model is called with a prompt asking it to classify the message as either SHOPPING_RELATED or OFF_TOPIC.

If the classification is OFF_TOPIC, the function returns False and the server immediately responds with a redirect message without calling the agent at all. This prevents prompt injection attacks where a user tries to override the system prompt by asking the shopping assistant to do something unrelated.

Image upload requests bypass the guardrail because the message is constructed by the server itself and always starts with a trusted prefix.

### Streaming Response

The server uses FastAPI's StreamingResponse with text/event-stream media type to implement Server-Sent Events. After the agent returns its full response, the server splits the text by spaces and streams each word back to the browser with a 30 millisecond delay between words. The browser accumulates the incoming chunks and re-renders the message content after each word arrives, creating a smooth typing animation without any client-side timers or polling.

### Dynamic System Prompt

The system prompt is not a static string. Every time the agent is invoked, the server fetches the user's preferences from PostgreSQL and appends them to the base system prompt as an ACTIVE USER PREFERENCES block. This means if a user saves a $10 budget in one session, the agent will automatically apply that constraint in every future session without the user having to mention it again.

---

## Project Structure

```
CartPilot/
    server.py              FastAPI application, SSE streaming, Redis session management
    shopping_agent.py      PersonalShopperAgent, ReAct tools, guardrail, dynamic system prompt
    reviews_api.py         Async helper for rating queries against the reviews table
    setup_db.py            One-time migration: creates PostgreSQL tables and seeds Pinecone
    requirements.txt       All Python dependencies
    render.yaml            Render deployment config with env var declarations
    public/
        index.html         Single-page app shell with favicon
        style.css          Full UI with light/dark mode, animations, skeleton loader
        script.js          SSE stream reader, localStorage session persistence, markdown rendering
```

---

## Cloud Services Used and Why

| Service | Purpose | Why I Chose It |
|---|---|---|
| Supabase | PostgreSQL database | Managed Postgres with a generous free tier, SQL-compatible, no infrastructure setup |
| Pinecone | Vector database | Industry standard for production vector search, serverless option with free tier |
| Upstash Redis | Session memory | HTTP-based Redis that works in ephemeral environments, no persistent connection needed |
| Groq | LLM and vision inference | Fastest available inference for open Llama models |
| Render | Hosting platform | Simple git-based deploys, free web service tier, reads render.yaml automatically |

---

## Features

**ReAct Agent with Tool Chaining** - The AI autonomously decides which tools to call and in what order. A single user message can trigger 5 or more tool calls chained together without any hardcoded logic.

**Hybrid Search** - SQL hard filtering combined with Pinecone semantic search. SQL handles exact constraints, vectors handle meaning. The intersection of both is more accurate than either alone.

**Multimodal Image Input** - Upload a product photo and the agent identifies it using a vision model and automatically searches for similar items in the catalog.

**Persistent Redis Memory** - Conversation history stored in Upstash Redis. Server restarts do not lose conversations. The application is horizontally scalable because no state is in the server process.

**Dynamic User Preferences** - Long-term preferences like budget limits or organic-only preferences are stored in PostgreSQL and injected into the system prompt automatically on every request.

**Input Guardrail Classifier** - LLM-based classifier runs before every agent invocation to block off-topic and prompt injection attempts.

**Word-by-Word Streaming** - FastAPI StreamingResponse sends the response word by word over SSE so the user sees a typing effect while the AI is processing.

**Async Architecture** - Every database call, every model call, and every server handler uses Python async/await. The server never blocks the event loop.

---

## Local Setup

### Step 1: Clone the repository and install dependencies

```bash
git clone https://github.com/YadavAkhileshh/CartPilot.git
cd CartPilot
pip install -r requirements.txt
```

### Step 2: Create your .env file

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres.xxxxx:yourpassword@aws-0-us-east-1.pooler.supabase.com:6543/postgres
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=cartpilot-products
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token
```

Note for the DATABASE_URL: if your Supabase password contains special characters like @ or # you must URL-encode them. Replace @ with %40 and # with %23.

### Step 3: Seed the cloud databases

```bash
python setup_db.py
```

This script creates the four tables in Supabase if they do not exist, truncates and re-inserts 32 products across 7 categories with review data, then reads all products from PostgreSQL, generates embeddings, and pushes them to Pinecone. Run this once before starting the server.

### Step 4: Start the server

```bash
python server.py
```

Open your browser at http://localhost:8000.

---

## Deploying to Render

The render.yaml file in this repository already defines everything Render needs.

1. Push the repository to GitHub.
2. Go to render.com and create a new Web Service linked to your GitHub repository.
3. Render will detect render.yaml and configure the build and start commands automatically.
4. In the Render dashboard under Environment Variables, add all six variables from your .env file.
5. Trigger a deploy.

The application is stateless by design. There are no local files that need to persist between deploys. All data lives in Supabase, Pinecone, and Upstash Redis. Cold starts on Render's free tier will be slow on first boot because the sentence-transformers model downloads at startup, but after that performance is normal.

---

## Design Decisions & Technical FAQ

**What is the ReAct pattern?** - ReAct stands for Reasoning and Acting. The model alternates between reasoning about what to do and calling a tool to do it. After each tool call it observes the result and reasons again. This repeats until the model can give a final answer. I use LangChain's create_agent which implements this loop.

**Why hybrid search instead of pure vector search?** - Vector search finds semantically similar products but cannot enforce a price constraint reliably. A query for "cheap honey" might match an expensive product because the word honey is semantically close. SQL handles the hard numeric and boolean filters first, then vector search handles the semantic ranking within that filtered set.

**What does stateless architecture mean?** - The server process holds no state between requests. All conversation history is in Redis, all product and order data is in PostgreSQL, all vectors are in Pinecone. You can restart the server, scale it to ten instances, or replace it entirely and every user will still get their full conversation history on the next request.

**Why asyncpg instead of psycopg2?** - psycopg2 is synchronous. Calling it inside an async FastAPI handler would block the entire event loop while waiting for the database, which means no other requests can be processed during that time. asyncpg is a pure async driver that releases the event loop while waiting for the database response.

**What are the guardrails doing?** - Before every request reaches the ReAct agent I run a separate LLM call that classifies the message as SHOPPING_RELATED or OFF_TOPIC. If it is off-topic the agent is never called. This prevents prompt injection where a user tries to override the system prompt, and it protects against users trying to use the shopping assistant as a general-purpose AI.

**How does the session memory work?** - The browser generates a UUID on first visit and stores it in localStorage. Every request sends this session ID to the server. The server uses it to fetch the conversation history from Redis at the start of the request and save the updated history at the end. The user gets the same session ID across page refreshes because localStorage persists in the browser.

---

## What I Learned Building This

The biggest lesson was the difference between a working prototype and a production-ready application. Replacing SQLite with Supabase and FAISS with Pinecone was not just a technical change. It forced me to think about async connection drivers, what happens when the process restarts, and how to design data persistence that is independent of any single server instance.

The second lesson was about the limits of pure vector search. Semantic search is powerful for finding relevant products but it cannot enforce a numeric constraint reliably. The hybrid approach where SQL handles the hard filters and vector search handles the semantic ranking turned out to be much more accurate than either approach alone.

The third lesson was about the ReAct pattern specifically. I had used LangChain before but I did not fully understand what the agent was doing internally. Implementing the guardrail wrapper forced me to understand the request flow deeply enough to intercept it before the ReAct loop starts. That is when it became clear how much work LangChain is doing behind the scenes on each call.
