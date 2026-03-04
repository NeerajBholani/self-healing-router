# How Different Systems Handle Tool Failures

The same scenario — customer support refund where Stripe is down — implemented across four approaches.

## The Scenario

A customer requests a refund. The agent needs to: look up order (CRM) -> process refund (Stripe or Razorpay) -> send confirmation (Email or SMS).

**Failure:** Stripe is down. What happens?

---

## 1. OpenAI Agents SDK (ReAct pattern)

```python
from openai import OpenAI
from agents import Agent, Runner, function_tool

client = OpenAI()

@function_tool
def process_refund_stripe(order_id: str, amount: float):
    """Process refund via Stripe."""
    raise ConnectionError("Stripe API is down")  # FAILS

@function_tool
def process_refund_razorpay(order_id: str, amount: float):
    """Process refund via Razorpay."""
    return {"status": "success", "provider": "razorpay"}

agent = Agent(
    name="Refund Agent",
    instructions="Process the customer's refund request.",
    tools=[process_refund_stripe, process_refund_razorpay],
)

# What happens:
# 1. LLM decides to try Stripe           -> LLM call #1
# 2. Stripe fails, error returned to LLM -> LLM call #2 (reason about failure)
# 3. LLM decides to try Razorpay         -> LLM call #3
# 4. Razorpay succeeds                   -> LLM call #4 (summarize result)
#
# Total: 4 LLM calls for ONE tool failure
# Cost: ~$0.04 at $0.01/call
# Latency: ~2-4 seconds (4 round trips)
# Correct: Yes
# Silent failures: 0 (LLM reasons about everything)
```

**Verdict:** Correct, but expensive. Every failure = more LLM calls. The LLM is doing routing work that doesn't need intelligence.

---

## 2. Anthropic Claude SDK (ReAct pattern)

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "process_refund_stripe",
        "description": "Process refund via Stripe",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"}
            }
        }
    },
    {
        "name": "process_refund_razorpay",
        "description": "Process refund via Razorpay (backup)",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"}
            }
        }
    }
]

messages = [{"role": "user", "content": "Process refund for order #1234, amount $50"}]

# Tool use loop
while True:
    response = client.messages.create(         # LLM call
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")

        # Execute the tool
        if tool_block.name == "process_refund_stripe":
            result = {"error": "Stripe API is down"}   # FAILS
        elif tool_block.name == "process_refund_razorpay":
            result = {"status": "success"}              # Works

        # Feed result back to LLM for next decision
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_block.id,
            "content": str(result)
        }]})
    else:
        break  # Done

# What happens:
# 1. LLM picks Stripe                    -> LLM call #1
# 2. Stripe fails, error fed back        -> LLM call #2 (reason about error)
# 3. LLM picks Razorpay                  -> LLM call #3
# 4. Razorpay succeeds, LLM summarizes   -> LLM call #4
#
# Same pattern as OpenAI: 4 LLM calls per single failure
```

**Verdict:** Same as OpenAI. Both SDKs use the LLM as the router. Correct but costly.

---

## 3. LangGraph (Static State Machine)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RefundState(TypedDict):
    order_id: str
    amount: float
    payment_result: str | None
    notification_result: str | None

def lookup_order(state):
    return {**state, "order_id": state["order_id"]}

def try_stripe(state):
    raise ConnectionError("Stripe is down")  # FAILS

def try_razorpay(state):
    return {**state, "payment_result": "razorpay_success"}

def send_email(state):
    return {**state, "notification_result": "email_sent"}

def route_payment(state):
    # Pre-coded: if Stripe fails, try Razorpay
    # But what if BOTH are down? No edge for that.
    if state.get("payment_result") is None:
        return "razorpay"
    return "email"

graph = StateGraph(RefundState)
graph.add_node("lookup", lookup_order)
graph.add_node("stripe", try_stripe)
graph.add_node("razorpay", try_razorpay)
graph.add_node("email", send_email)

graph.set_entry_point("lookup")
graph.add_edge("lookup", "stripe")
graph.add_conditional_edges("stripe", route_payment,
    {"razorpay": "razorpay", "email": "email"})
graph.add_edge("razorpay", "email")
graph.add_edge("email", END)

# What happens with Stripe down:
# - Stripe fails -> route_payment returns "razorpay" -> works
# - 0 LLM calls, fast
#
# What happens with Stripe AND Razorpay down:
# - Stripe fails -> route_payment returns "razorpay"
# - Razorpay fails -> ??? No edge coded for this
# - State machine reaches END without payment
# - SILENT FAILURE: no error raised, customer thinks refund processed
#
# Total: 0 LLM calls
# Silent failures: 1 (on compound failure)
```

**Verdict:** Fast and cheap, but SILENT FAILURES on compound scenarios. The state machine has no edge for "both payment providers down" — it just proceeds without payment.

---

## 4. Self-Healing Router (This Paper)

```python
from src.tool_graph import ToolGraph
from src.monitors import MonitorBank
from src.orchestrator import Orchestrator

# Define the tool graph once
graph = ToolGraph()
for name in ['Start', 'CRM', 'Stripe', 'Razorpay', 'Email', 'SMS', 'Complete']:
    graph.add_node(name)
graph.add_edge('Start', 'CRM', 0.1)
graph.add_edge('CRM', 'Stripe', 0.3)
graph.add_edge('CRM', 'Razorpay', 0.5)
graph.add_edge('Stripe', 'Email', 0.3)
graph.add_edge('Razorpay', 'Email', 0.3)
graph.add_edge('Stripe', 'SMS', 0.4)
graph.add_edge('Razorpay', 'SMS', 0.4)
graph.add_edge('Email', 'Complete', 0.1)
graph.add_edge('SMS', 'Complete', 0.1)

monitors = MonitorBank()
orchestrator = Orchestrator(graph=graph, monitors=monitors)

# Stripe down? Dijkstra finds: Start -> CRM -> Razorpay -> Email -> Complete
# Stripe AND Razorpay down? Dijkstra returns null -> escalate to LLM
# Stripe AND Email down? Dijkstra finds: Start -> CRM -> Razorpay -> SMS -> Complete

result = orchestrator.execute_task(
    goal="process_refund",
    start="Start",
    target="Complete",
    request={"text": "refund order #1234"}
)

# What happens with Stripe down:
# - Health monitor detects Stripe down
# - Edges to/from Stripe set to infinity
# - Dijkstra finds: CRM -> Razorpay -> Email -> Complete
# - 0 LLM calls, sub-millisecond
#
# What happens with Stripe AND Razorpay down:
# - Both reweighted to infinity
# - Dijkstra returns null (no path)
# - LLM invoked ONCE for goal demotion
# - EXPLICITLY LOGGED as escalation
#
# Total: 0 LLM calls (single failure), 1 LLM call (no-path escalation)
# Silent failures: 0 (by construction)
```

**Verdict:** Combines the best of both. Fast like LangGraph (0 LLM calls when alternatives exist). Correct like ReAct (never silently skips). Escalates to LLM only when mathematically necessary (no path exists).

---

## Summary

| | Stripe down | Stripe + Razorpay down | Stripe + Email down |
|---|---|---|---|
| **OpenAI SDK** | 4 LLM calls, correct | 6 LLM calls, correct | 6 LLM calls, correct |
| **Claude SDK** | 4 LLM calls, correct | 6 LLM calls, correct | 6 LLM calls, correct |
| **LangGraph** | 0 LLM calls, correct | 0 LLM calls, **SILENT FAILURE** | 0 LLM calls, correct |
| **Self-Healing Router** | 0 LLM calls, correct | 1 LLM call, correct (escalated) | 0 LLM calls, correct (rerouted) |

The key insight: OpenAI/Claude SDKs use the LLM for every routing decision. LangGraph removes the LLM but loses correctness on compound failures where both a tool and its fallback are down. Self-Healing Router removes the LLM from routing while preserving correctness — and escalates only when no alternative exists.
