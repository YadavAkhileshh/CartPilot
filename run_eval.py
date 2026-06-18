import sqlite3
import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from setup_db import create_database
from shopping_agent import agent, DB_PATH, llm
from langchain_core.messages import HumanMessage, AIMessage

def reset_database():
    """Reset database to a clean default state with no orders or preferences."""
    print("\n--- Resetting Database ---")
    create_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS user_preferences")
    cursor.execute("CREATE TABLE user_preferences (key TEXT PRIMARY KEY, value TEXT)")
    cursor.execute("DELETE FROM orders")
    conn.commit()
    conn.close()
    print("Database reset completed.")

def agent_invoke(input_data: dict) -> dict:
    """Wrapper around agent.invoke with built-in sleep to avoid rate limits."""
    print("Waiting 6 seconds before agent call to avoid rate limits...")
    time.sleep(6)
    return agent.invoke(input_data)

def get_tool_calls(messages: List[Any], tool_name: str = None) -> List[Dict[str, Any]]:
    """Extract tool calls of a given name from the agent message list."""
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tool_name is None or tc.get("name") == tool_name:
                    calls.append(tc)
    return calls

def llm_judge(query: str, response: str) -> Dict[str, Any]:
    """Use an LLM-as-judge to evaluate response quality."""
    prompt = f"""You are an expert AI judge evaluating a shopping assistant's response.
Evaluate the following interaction and score it on three criteria on a scale of 0 to 5 (where 0 is completely incorrect/failing, and 5 is perfect).

User Query: "{query}"
Shopping Assistant Response:
---
{response}
---

Criteria for Evaluation:

1. RELEVANCE (Score 0 to 5):
- Did the assistant directly address the user's request?
- Is there any off-topic or irrelevant information?
- For off-topic requests (like asking for weather or poems), did the assistant politely redirect the user without answering?

2. CORRECTNESS (Score 0 to 5):
- Did the assistant return the correct products matching the user's constraints (e.g. organic, price, rating)?
- If the user ordered a product, did it confirm the correct product and ID?
- If the user asked for order history or preferences, is the answer correct?

3. FORMAT COMPLIANCE (Score 0 to 5):
- If products are listed, they MUST use this exact plain text format:
  #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>
- No bold (**), no italic (*), no code blocks (```), and no backticks (`) should be used in the product list lines.
- There should be a blank line between each product entry.
- Give a 5 if format is fully met. Give a 0 if the list format is ignored.

Please output your evaluation strictly in JSON format with the following keys:
- relevance_score (int)
- relevance_reason (str)
- correctness_score (int)
- correctness_reason (str)
- format_compliance_score (int)
- format_compliance_reason (str)
- overall_summary (str)

Do not include any markdown fences or other text besides the raw JSON."""

    print("Waiting 6 seconds before LLM judge call to avoid rate limits...")
    time.sleep(6)
    res = llm.invoke(prompt)
    content = res.content.strip()
    if "<think>" in content.lower():
        parts = content.lower().split("</think>")
        if len(parts) > 1:
            content = parts[-1].strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing judge output: {e}\nRaw output: {content}")
        return {
            "relevance_score": 0,
            "relevance_reason": f"Failed to parse judge output: {e}",
            "correctness_score": 0,
            "correctness_reason": "Failed to parse judge output",
            "format_compliance_score": 0,
            "format_compliance_reason": "Failed to parse judge output",
            "overall_summary": content
        }

def run_tests():
    reset_database()
    
    results = []
    
    # -------------------------------------------------------------------------
    # Test 1: Tool Call Accuracy - Search Organic Honey under $20
    # -------------------------------------------------------------------------
    print("\n--- Running Test 1: Search Organic Honey under $20 ---")
    query_1 = "I want to buy organic honey under $20"
    r1 = agent_invoke({"messages": [{"role": "user", "content": query_1}]})
    msg_list_1 = r1["messages"]
    final_resp_1 = msg_list_1[-1].content
    print(f"Agent response:\n{final_resp_1}")
    
    # Assert tool call
    search_calls = get_tool_calls(msg_list_1, "search_products")
    t1_tool_ok = len(search_calls) > 0
    t1_args_ok = False
    if t1_tool_ok:
        args = search_calls[0].get("args", {})
        # supports both boolean and string representations
        is_organic = args.get("is_organic")
        max_price = args.get("max_price")
        t1_args_ok = (is_organic is True or str(is_organic).lower() == "true") and float(max_price) == 20.0
        print(f"Tool call found: {search_calls[0]['name']} with args {args}")
    else:
        print("Error: No search_products tool call found.")

    # LLM Judge evaluation
    judge_1 = llm_judge(query_1, final_resp_1)
    print(f"Judge Evaluation:\n{json.dumps(judge_1, indent=2)}")
    
    results.append({
        "test_name": "Test 1: Search Organic Honey under $20",
        "tool_call_ok": t1_tool_ok and t1_args_ok,
        "tool_call_details": f"Called search_products: {t1_tool_ok}, Args correct: {t1_args_ok}",
        "judge_relevance": judge_1.get("relevance_score"),
        "judge_correctness": judge_1.get("correctness_score"),
        "judge_format": judge_1.get("format_compliance_score"),
        "judge_summary": judge_1.get("overall_summary")
    })

    # -------------------------------------------------------------------------
    # Test 2: Input Guardrail - Off-Topic Rejection
    # -------------------------------------------------------------------------
    print("\n--- Running Test 2: Input Guardrail - Off-Topic Rejection ---")
    query_2 = "write me a poem about bees"
    r2 = agent_invoke({"messages": [{"role": "user", "content": query_2}]})
    msg_list_2 = r2["messages"]
    final_resp_2 = msg_list_2[-1].content
    print(f"Agent response:\n{final_resp_2}")
    
    # Assert tool call - expect none
    all_calls = get_tool_calls(msg_list_2)
    t2_tool_ok = len(all_calls) == 0
    if not t2_tool_ok:
        print(f"Error: Unexpected tool calls found: {all_calls}")
    else:
        print("Success: No tools called for off-topic query.")

    judge_2 = llm_judge(query_2, final_resp_2)
    print(f"Judge Evaluation:\n{json.dumps(judge_2, indent=2)}")

    results.append({
        "test_name": "Test 2: Input Guardrail - Off-Topic Rejection",
        "tool_call_ok": t2_tool_ok,
        "tool_call_details": "No tools called: True" if t2_tool_ok else f"Tools called: {all_calls}",
        "judge_relevance": judge_2.get("relevance_score"),
        "judge_correctness": judge_2.get("correctness_score"),
        "judge_format": judge_2.get("format_compliance_score"),
        "judge_summary": judge_2.get("overall_summary")
    })

    # -------------------------------------------------------------------------
    # Test 3: Preferences - Save and Apply
    # -------------------------------------------------------------------------
    print("\n--- Running Test 3: Preferences - Save and Apply ---")
    # Start a conversation thread/session
    session_messages = []
    
    # 3a. Save organic preference
    query_3a = "remember that I always prefer organic products"
    session_messages.append({"role": "user", "content": query_3a})
    r3a = agent_invoke({"messages": session_messages})
    final_resp_3a = r3a["messages"][-1].content
    print(f"Agent response (3a):\n{final_resp_3a}")
    session_messages.append({"role": "assistant", "content": final_resp_3a})
    
    save_calls = get_tool_calls(r3a["messages"], "save_user_preference")
    t3a_tool_ok = len(save_calls) > 0
    t3a_args_ok = False
    if t3a_tool_ok:
        args = save_calls[0].get("args", {})
        t3a_args_ok = args.get("key") == "prefers_organic" and args.get("value") in ["True", "true"]
        print(f"Tool call found: {save_calls[0]['name']} with args {args}")
    else:
        print("Error: No save_user_preference tool call found.")

    # 3b. Search honey and verify preference is applied automatically
    query_3b = "show me honey"
    session_messages.append({"role": "user", "content": query_3b})
    r3b = agent_invoke({"messages": session_messages})
    final_resp_3b = r3b["messages"][-1].content
    print(f"Agent response (3b):\n{final_resp_3b}")
    session_messages.append({"role": "assistant", "content": final_resp_3b})

    search_calls_3b = get_tool_calls(r3b["messages"], "search_products")
    t3b_tool_ok = len(search_calls_3b) > 0
    t3b_args_ok = False
    if t3b_tool_ok:
        # Check that is_organic=True is automatically passed
        args = search_calls_3b[0].get("args", {})
        t3b_args_ok = args.get("is_organic") is True or str(args.get("is_organic")).lower() == "true"
        print(f"Tool call found (3b): {search_calls_3b[0]['name']} with args {args}")
    else:
        print("Error: No search_products tool call found in 3b.")

    judge_3 = llm_judge(query_3b, final_resp_3b)
    print(f"Judge Evaluation (3b):\n{json.dumps(judge_3, indent=2)}")

    results.append({
        "test_name": "Test 3: Preferences - Save and Apply",
        "tool_call_ok": t3a_tool_ok and t3a_args_ok and t3b_tool_ok and t3b_args_ok,
        "tool_call_details": f"Save preference ok: {t3a_args_ok}, Search applied preference ok: {t3b_args_ok}",
        "judge_relevance": judge_3.get("relevance_score"),
        "judge_correctness": judge_3.get("correctness_score"),
        "judge_format": judge_3.get("format_compliance_score"),
        "judge_summary": judge_3.get("overall_summary")
    })

    # -------------------------------------------------------------------------
    # Test 4: Checkout and Order History Summary
    # -------------------------------------------------------------------------
    print("\n--- Running Test 4: Checkout and Order History Summary ---")
    session_4 = []
    
    # 4a. Browse organic honey under $20
    query_4a = "organic honey under $20"
    session_4.append({"role": "user", "content": query_4a})
    r4a = agent_invoke({"messages": session_4})
    final_resp_4a = r4a["messages"][-1].content
    print(f"Agent response (4a):\n{final_resp_4a}")
    session_4.append({"role": "assistant", "content": final_resp_4a})
    
    # 4b. Confirm ordering the first one (ID 1)
    query_4b = "yes, order the first one"
    session_4.append({"role": "user", "content": query_4b})
    r4b = agent_invoke({"messages": session_4})
    final_resp_4b = r4b["messages"][-1].content
    print(f"Agent response (4b):\n{final_resp_4b}")
    session_4.append({"role": "assistant", "content": final_resp_4b})

    checkout_calls = get_tool_calls(r4b["messages"], "checkout")
    t4b_tool_ok = len(checkout_calls) > 0
    t4b_args_ok = False
    if t4b_tool_ok:
        args = checkout_calls[0].get("args", {})
        t4b_args_ok = int(args.get("product_id")) == 1
        print(f"Tool call found: {checkout_calls[0]['name']} with args {args}")
    else:
        print("Error: No checkout tool call found.")

    # 4c. Query order history
    query_4c = "what did I order before?"
    session_4.append({"role": "user", "content": query_4c})
    r4c = agent_invoke({"messages": session_4})
    final_resp_4c = r4c["messages"][-1].content
    print(f"Agent response (4c):\n{final_resp_4c}")
    session_4.append({"role": "assistant", "content": final_resp_4c})

    history_calls = get_tool_calls(r4c["messages"], "get_order_history")
    t4c_tool_ok = len(history_calls) > 0
    print(f"Order history tool call found: {t4c_tool_ok}")

    judge_4 = llm_judge(query_4c, final_resp_4c)
    print(f"Judge Evaluation (4c):\n{json.dumps(judge_4, indent=2)}")

    results.append({
        "test_name": "Test 4: Checkout and Order History Summary",
        "tool_call_ok": t4b_tool_ok and t4b_args_ok and t4c_tool_ok,
        "tool_call_details": f"Checkout ID 1 ok: {t4b_args_ok}, History called: {t4c_tool_ok}",
        "judge_relevance": judge_4.get("relevance_score"),
        "judge_correctness": judge_4.get("correctness_score"),
        "judge_format": judge_4.get("format_compliance_score"),
        "judge_summary": judge_4.get("overall_summary")
    })

    # Print summary report
    print("\n=======================================================")
    print("                 EVALUATION REPORT")
    print("=======================================================")
    all_ok = True
    for r in results:
        print(f"\n{r['test_name']}:")
        print(f"  - Tool Call Accuracy: {'PASS' if r['tool_call_ok'] else 'FAIL'} ({r['tool_call_details']})")
        print(f"  - Judge Scores: Relevance={r['judge_relevance']}/5, Correctness={r['judge_correctness']}/5, Format={r['judge_format']}/5")
        print(f"  - Judge Summary: {r['judge_summary']}")
        if not r['tool_call_ok'] or r['judge_relevance'] < 4 or r['judge_correctness'] < 4:
            all_ok = False
            
    print("\n=======================================================")
    if all_ok:
        print("OVERALL RESULT: ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL RESULT: SOME TESTS FAILED.")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
