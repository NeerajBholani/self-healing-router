# Self-Healing Router

**Deterministic tool routing for cost-efficient LLM agents.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.01548-b31b1b.svg)](https://arxiv.org/abs/2603.01548)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

When an LLM agent's tool fails, should you ask the LLM what to do (expensive) or pre-code every fallback (brittle)? Neither. Use a graph algorithm.

Self-Healing Router treats agent control-flow decisions as **routing, not reasoning**. It uses Dijkstra's shortest-path algorithm on a cost-weighted tool graph with parallel health monitors. When a tool fails mid-execution, its edges are reweighted to infinity and the path is recomputed -- yielding automatic recovery in sub-millisecond time without invoking the LLM.

**Paper:** [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548) (arXiv:2603.01548)

## Key Results (19 benchmark scenarios)

| System | Correct | LLM Calls | Silent Failures |
|--------|---------|-----------|-----------------|
| **Self-Healing Router** | **19/19** | **9** | **0** |
| ReAct | 19/19 | 123 | 0 |
| LangGraph (well-eng.) | 16/19 | 0 | 3 |

- **93% fewer LLM calls** than ReAct (9 vs 123)
- **Zero silent failures** -- every failure is a logged reroute or explicit LLM escalation
- Matches ReAct correctness at LangGraph-like cost

## Architecture

```
Parallel Health       Orchestrator        Tool Graph
  Monitors                                (Dijkstra)

 Intent  (0.90)     1. max(score)       Start --> Stripe --+
 Risk    (0.95) --> 2. Reweight    -->    |      (fail)    |
 Health  (0.99)     3. Dijkstra          +--> Razorpay ----+
 Progress(0.50)     4. or --> LLM              +--> Goal
                          |
                    LLM (rare)
                    Goal demotion
```

## Quick Start

```bash
git clone https://github.com/NeerajBholani/self-healing-router.git
cd self-healing-router

# Run the full benchmark (reproduces paper results)
python -m tests.run_evaluation

# Interactive demo -- break tools and watch Dijkstra reroute
python -m demo.interactive
```

**No dependencies required** -- pure Python 3.10+, standard library only.

## Web Demo

**[Try it live in your browser](https://NeerajBholani.github.io/self-healing-router/)** — break tools, watch Dijkstra reroute, and compare against LangGraph's static fallbacks on compound failure scenarios.

## CLI Demo

```
$ python -m demo.interactive

Choose a scenario:
  [1] Customer Support (Stripe/Razorpay -> Email/SMS)
  [2] Travel Booking (Flight/Train -> Hotel/Hostel -> Car/Bus)

> fail stripe
  Stripe is now DOWN
  Path: Start -> CRM -> Razorpay -> Email -> Complete (cost: 1.00)

> fail email
  Email is now DOWN
  Path: Start -> CRM -> Razorpay -> SMS -> Complete (cost: 1.10)

> fail razorpay
  Razorpay is now DOWN
  Path: NO PATH EXISTS (LLM escalation required)
```

## Project Structure

```
self-healing-router/
├── src/
│   ├── tool_graph.py      # Dijkstra + dynamic edge reweighting
│   ├── monitors.py        # Parallel health monitors
│   └── orchestrator.py    # Self-healing execution loop
├── scenarios/
│   └── benchmarks.py      # 19 benchmark scenarios (3 domains)
├── baselines/
│   └── baselines.py       # ReAct and LangGraph baselines
├── tests/
│   └── run_evaluation.py  # Full benchmark runner
├── demo/
│   ├── interactive.py     # Interactive CLI demo
│   └── web_demo.jsx       # React web demo (compare SHR vs LangGraph)
└── README.md
```

## How It Works

### The Four-Step Recovery Sequence

When a tool fails mid-execution:

1. **Detect** -- Tool returns error (or health monitor fires pre-emptively)
2. **Reweight** -- Set failed tool edges to infinity (O(degree), typically 2-4 edges)
3. **Recompute** -- Dijkstra finds cheapest alternative path (sub-millisecond)
4. **Resume** -- Continue execution on new path

If Dijkstra returns null (no path exists), escalate to LLM for goal demotion.

### Binary Observability Guarantee

Every failure results in exactly one of two outcomes:
- **Logged reroute** -- Dijkstra found an alternative path (recorded in decision log)
- **Explicit LLM escalation** -- No path exists, LLM invoked for goal demotion (recorded)

There is no third state. A tool failure can never silently pass through the system.

### Why Not Just Use the LLM?

For 8 tools with binary failure (up/down), there are 2^8 = 256 possible failure combinations. Dijkstra handles all 256 with a single algorithm call. ReAct uses an LLM call per decision.

## SDK Comparison

See **[COMPARISON.md](COMPARISON.md)** for the same refund scenario implemented across OpenAI Agents SDK, Claude SDK, LangGraph, and Self-Healing Router — showing exactly where each approach breaks down.

## Citation

```bibtex
@article{bholani2026selfhealing,
  title={Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents},
  author={Bholani, Neeraj},
  journal={arXiv preprint arXiv:2603.01548},
  year={2026}
}
```

## License

MIT
