#!/usr/bin/env python3
"""
Interactive Self-Healing Router Demo.

Lets you:
1. Pick a scenario (customer support, travel, content moderation)
2. Break tools in real-time
3. Watch Dijkstra reroute automatically
4. See the observability log

Usage:
    python -m demo.interactive
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tool_graph import ToolGraph
from src.monitors import MonitorBank
from src.orchestrator import Orchestrator, ExecutionResult


def build_customer_support_graph() -> ToolGraph:
    """Build the customer support example from the paper."""
    g = ToolGraph()
    for name in ['Start', 'CRM', 'Stripe', 'Razorpay', 'Email', 'SMS', 'Complete']:
        g.add_node(name)
    g.add_edge('Start', 'CRM', 0.1)
    g.add_edge('CRM', 'Stripe', 0.3)
    g.add_edge('CRM', 'Razorpay', 0.5)
    g.add_edge('Stripe', 'Email', 0.3)
    g.add_edge('Stripe', 'SMS', 0.4)
    g.add_edge('Razorpay', 'Email', 0.3)
    g.add_edge('Razorpay', 'SMS', 0.4)
    g.add_edge('Email', 'Complete', 0.1)
    g.add_edge('SMS', 'Complete', 0.1)
    return g


def build_travel_graph() -> ToolGraph:
    """Build the travel booking example."""
    g = ToolGraph()
    for name in ['Start', 'FlightAPI', 'TrainAPI', 'HotelAPI', 'HostelAPI',
                  'CarAPI', 'BusAPI', 'Payment', 'Confirm']:
        g.add_node(name)
    g.add_edge('Start', 'FlightAPI', 0.3)
    g.add_edge('Start', 'TrainAPI', 0.5)
    g.add_edge('FlightAPI', 'HotelAPI', 0.3)
    g.add_edge('FlightAPI', 'HostelAPI', 0.4)
    g.add_edge('TrainAPI', 'HotelAPI', 0.3)
    g.add_edge('TrainAPI', 'HostelAPI', 0.4)
    g.add_edge('HotelAPI', 'CarAPI', 0.3)
    g.add_edge('HotelAPI', 'BusAPI', 0.4)
    g.add_edge('HostelAPI', 'CarAPI', 0.4)
    g.add_edge('HostelAPI', 'BusAPI', 0.3)
    g.add_edge('CarAPI', 'Payment', 0.2)
    g.add_edge('BusAPI', 'Payment', 0.2)
    g.add_edge('Payment', 'Confirm', 0.1)
    return g


def print_graph_state(graph: ToolGraph, title: str = ""):
    """Pretty-print the current graph state."""
    if title:
        print(f"\n  📊 {title}")
    print(f"  Nodes: {', '.join(n.name for n in graph.nodes.values())}")
    healthy = [n.name for n in graph.nodes.values() if n.healthy]
    failed = [n.name for n in graph.nodes.values() if not n.healthy]
    print(f"  Healthy: {', '.join(healthy) or 'none'}")
    if failed:
        print(f"  Failed:  {', '.join(failed)} ❌")


def print_path(graph: ToolGraph, start: str, goal: str):
    """Show the current shortest path."""
    result = graph.dijkstra(start, goal)
    if result:
        path, cost, ns = result
        print(f"  Path: {' → '.join(path)} (cost: {cost:.2f}, computed in {ns}ns)")
    else:
        print(f"  Path: NO PATH EXISTS ⚠️  (LLM escalation required)")


def interactive_demo():
    """Run the interactive demo."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Self-Healing Router — Interactive Demo          ║")
    print("║     Break tools and watch Dijkstra reroute!         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("Choose a scenario:")
    print("  [1] Customer Support (Stripe/Razorpay → Email/SMS)")
    print("  [2] Travel Booking (Flight/Train → Hotel/Hostel → Car/Bus)")
    print("  [3] Run full benchmark (19 scenarios)")
    print("  [q] Quit")
    
    choice = input("\n> ").strip()
    
    if choice == '3':
        from tests.run_evaluation import run_evaluation
        run_evaluation()
        return
    
    if choice == 'q':
        return
    
    if choice == '1':
        graph = build_customer_support_graph()
        start, goal = 'Start', 'Complete'
        tools = ['CRM', 'Stripe', 'Razorpay', 'Email', 'SMS']
        scenario_name = "Customer Support Refund"
    elif choice == '2':
        graph = build_travel_graph()
        start, goal = 'Start', 'Confirm'
        tools = ['FlightAPI', 'TrainAPI', 'HotelAPI', 'HostelAPI', 'CarAPI', 'BusAPI', 'Payment']
        scenario_name = "Travel Booking"
    else:
        print("Invalid choice")
        return
    
    monitors = MonitorBank()
    failed_tools = set()
    
    print(f"\n{'='*55}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*55}")
    print_graph_state(graph, "Initial State")
    print_path(graph, start, goal)
    
    print(f"\nCommands:")
    print(f"  fail <tool>    — Take a tool offline")
    print(f"  heal <tool>    — Bring a tool back online")
    print(f"  path           — Show current shortest path")
    print(f"  run            — Execute the task end-to-end")
    print(f"  reset          — Reset all tools to healthy")
    print(f"  quit           — Exit")
    print(f"\n  Available tools: {', '.join(tools)}")
    
    while True:
        cmd = input(f"\n🔧 > ").strip().lower()
        
        if cmd == 'quit' or cmd == 'q':
            break
        
        elif cmd.startswith('fail '):
            tool = cmd[5:].strip()
            # Case-insensitive matching
            match = next((t for t in tools if t.lower() == tool.lower()), None)
            if match:
                graph.reweight_failed_node(match)
                monitors.tool_health.mark_down(match)
                failed_tools.add(match)
                print(f"  ❌ {match} is now DOWN")
                print_path(graph, start, goal)
            else:
                print(f"  Unknown tool. Available: {', '.join(tools)}")
        
        elif cmd.startswith('heal '):
            tool = cmd[5:].strip()
            match = next((t for t in tools if t.lower() == tool.lower()), None)
            if match:
                graph.restore_node(match)
                monitors.tool_health.mark_up(match)
                failed_tools.discard(match)
                print(f"  ✅ {match} is back ONLINE")
                print_path(graph, start, goal)
            else:
                print(f"  Unknown tool. Available: {', '.join(tools)}")
        
        elif cmd == 'path':
            print_graph_state(graph, "Current State")
            print_path(graph, start, goal)
        
        elif cmd == 'run':
            print(f"\n  🚀 Executing task: {scenario_name}")
            print(f"  Failed tools: {', '.join(failed_tools) or 'none'}")
            print()
            
            # Build fresh orchestrator
            def executor(tool_name, request):
                if tool_name in failed_tools:
                    print(f"    ❌ {tool_name} — FAILED")
                    return ExecutionResult(tool_name=tool_name, success=False,
                                         error=f"{tool_name} unavailable")
                print(f"    ✅ {tool_name} — SUCCESS")
                return ExecutionResult(tool_name=tool_name, success=True, output={})
            
            # Reset graph for fresh run (but keep failures)
            for edge in graph.edges:
                src_healthy = graph.nodes[edge.source].healthy
                tgt_healthy = graph.nodes[edge.target].healthy
                if src_healthy and tgt_healthy:
                    edge.current_cost = edge.base_cost
            
            orchestrator = Orchestrator(
                graph=graph,
                monitors=monitors,
                tool_executor=executor,
            )
            
            result = orchestrator.execute_task(
                goal=scenario_name,
                start=start,
                target=goal,
                request={"text": scenario_name}
            )
            
            print(f"\n  {'─'*45}")
            print(f"  Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
            print(f"  Path taken: {' → '.join(result.path_taken) or 'none'}")
            print(f"  LLM calls: {result.llm_calls}")
            print(f"  Graph reroutes: {result.graph_reroutes}")
            if result.demoted_goal:
                print(f"  Demoted goal: {result.demoted_goal}")
            
            print(f"\n  📋 Decision log:")
            for d in result.decisions:
                icon = {"reroute": "🔄", "escalate": "🆘", "execute": "▶️", "complete": "✅"}.get(d.action, "•")
                print(f"    {icon} [{d.action}] {d.detail}")
        
        elif cmd == 'reset':
            graph.reset()
            monitors.reset()
            failed_tools.clear()
            print("  🔄 All tools reset to healthy")
            print_path(graph, start, goal)
        
        else:
            print("  Commands: fail <tool>, heal <tool>, path, run, reset, quit")


if __name__ == "__main__":
    interactive_demo()
