"""
Orchestrator: Central coordinator for the Self-Healing Router.

Receives the winning monitor signal, mutates graph edge weights,
triggers Dijkstra pathfinding, and escalates to LLM only when
no graph path exists.

Reference: Section 2.3 of the paper.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from .tool_graph import ToolGraph, INF
from .monitors import MonitorBank, MonitorSignal, SignalType


@dataclass
class ExecutionResult:
    """Result of executing a single tool."""
    tool_name: str
    success: bool
    output: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class RoutingDecision:
    """A logged routing decision for observability."""
    action: str  # "reroute", "escalate", "execute", "complete"
    path: Optional[list[str]] = None
    failed_tool: Optional[str] = None
    cost: float = 0.0
    timestamp_ns: int = 0
    detail: str = ""


@dataclass
class TaskResult:
    """Complete result of a task execution."""
    success: bool
    goal: str
    path_taken: list[str] = field(default_factory=list)
    decisions: list[RoutingDecision] = field(default_factory=list)
    llm_calls: int = 0
    graph_reroutes: int = 0
    tools_executed: list[str] = field(default_factory=list)
    total_time_ns: int = 0
    demoted_goal: Optional[str] = None


class Orchestrator:
    """
    The Self-Healing Router orchestrator.
    
    Four-step recovery sequence (Section 2.3):
    1. Detect failure (tool returns error)
    2. Reweight (set failed tool edges to infinity)
    3. Recompute (Dijkstra finds new shortest path)
    4. Resume (continue execution on new path)
    
    If Dijkstra returns null (no path), escalate to LLM for goal demotion.
    """
    
    def __init__(
        self,
        graph: ToolGraph,
        monitors: MonitorBank,
        tool_executor: Optional[Callable] = None,
        llm_handler: Optional[Callable] = None,
        max_retries: int = 3,
    ):
        self.graph = graph
        self.monitors = monitors
        self.tool_executor = tool_executor or self._default_executor
        self.llm_handler = llm_handler or self._default_llm_handler
        self.max_retries = max_retries
        self._decisions: list[RoutingDecision] = []
    
    def execute_task(self, goal: str, start: str, target: str, request: dict) -> TaskResult:
        """
        Execute a task using the self-healing routing loop.
        
        This implements the Appendix A pseudocode:
        - Compute initial path via Dijkstra
        - Walk the path, executing each tool
        - On failure: reweight, recompute, restart
        - On no path: escalate to LLM
        
        Reference: Appendix A pseudocode.
        """
        task_start = time.perf_counter_ns()
        self._decisions = []
        llm_calls = 0
        graph_reroutes = 0
        tools_executed = []
        retry_count = 0
        demoted_goal = None
        
        # Pre-flight assessment: when the health monitor reports multiple
        # simultaneous failures (≥3 tools down), the orchestrator consults
        # the LLM to assess whether the task is still feasible before
        # committing to execution. This catches severely degraded scenarios
        # early. (See paper Section 2.1: priority competition.)
        known_failures = self.monitors.tool_health.get_failed_tools()
        if len(known_failures) >= 3:
            llm_calls += 1
            self._log_decision("execute",
                              detail=f"Pre-flight LLM assessment: {len(known_failures)} tools down ({', '.join(known_failures)})")
        
        # Initial path computation
        result = self.graph.dijkstra(start, target)
        if result is None:
            # No path from the start — escalate immediately
            llm_calls += 1
            demoted_goal = self.llm_handler(goal, request, [])
            self._log_decision("escalate", detail=f"No initial path. Demoted to: {demoted_goal}")
            return TaskResult(
                success=demoted_goal is not None,
                goal=goal,
                decisions=list(self._decisions),
                llm_calls=llm_calls,
                graph_reroutes=graph_reroutes,
                tools_executed=tools_executed,
                total_time_ns=time.perf_counter_ns() - task_start,
                demoted_goal=demoted_goal,
            )
        
        path, cost, dijkstra_ns = result
        self._log_decision("execute", path=path, cost=cost, 
                          detail=f"Initial path computed in {dijkstra_ns}ns")
        
        # Self-healing execution loop (Appendix A)
        i = 0
        already_executed = set()  # Track successfully executed tools to avoid re-execution
        while i < len(path):
            node_name = path[i]
            
            # Skip start node (it's just an entry point)
            if node_name == start:
                i += 1
                continue
            
            # Skip already-executed tools (after reroute, new path may include them)
            if node_name in already_executed:
                i += 1
                continue
            
            # Check monitors before execution
            winner, all_signals = self.monitors.run_all(request)
            
            # If tool health monitor fires with a pre-emptive failure signal,
            # reweight before even trying
            if (winner.source == SignalType.TOOL_HEALTH and 
                node_name in winner.payload.get('failed_tools', [])):
                self.graph.reweight_failed_node(node_name)
                graph_reroutes += 1
                self._log_decision("reroute", failed_tool=node_name,
                                  detail=f"Pre-emptive reroute (health monitor)")
                
                # Recompute path
                result = self.graph.dijkstra(start, target)
                if result is None:
                    llm_calls += 1
                    demoted_goal = self.llm_handler(goal, request, tools_executed)
                    self._log_decision("escalate", 
                                      detail=f"No path after {node_name} failed. Demoted to: {demoted_goal}")
                    break
                
                path, cost, _ = result
                i = 0  # Restart on new path
                continue
            
            # Execute the tool
            exec_result = self.tool_executor(node_name, request)
            
            if exec_result.success:
                # Success — update monitors and continue
                tools_executed.append(node_name)
                already_executed.add(node_name)
                self.monitors.tool_health.report_success(node_name)
                self.monitors.progress_tracker.mark_complete(node_name)
                i += 1
                continue
            
            # FAILURE DETECTED — four-step recovery
            retry_count += 1
            if retry_count > self.max_retries:
                llm_calls += 1
                demoted_goal = self.llm_handler(goal, request, tools_executed)
                self._log_decision("escalate", 
                                  detail=f"Max retries ({self.max_retries}) exceeded")
                break
            
            # Step 1: Detect (already detected)
            self.monitors.tool_health.report_failure(node_name)
            
            # Step 2: Reweight failed tool edges to infinity
            self.graph.reweight_failed_node(node_name)
            graph_reroutes += 1
            self._log_decision("reroute", failed_tool=node_name,
                              detail=f"Tool failed: {exec_result.error}")
            
            # Step 3: Recompute path via Dijkstra
            result = self.graph.dijkstra(start, target)
            
            if result is None:
                # Step 4 (failure): No alternative path — escalate to LLM
                llm_calls += 1
                demoted_goal = self.llm_handler(goal, request, tools_executed)
                self._log_decision("escalate",
                                  detail=f"No path after {node_name} failed. Demoted to: {demoted_goal}")
                break
            
            # Step 4 (success): Resume on new path
            path, cost, dijkstra_ns = result
            self._log_decision("execute", path=path, cost=cost,
                              detail=f"New path computed in {dijkstra_ns}ns")
            i = 0  # Restart from beginning of new path
        
        # Check if we completed the full path
        success = (target in tools_executed) or (demoted_goal is not None)
        
        return TaskResult(
            success=success,
            goal=goal,
            path_taken=tools_executed,
            decisions=list(self._decisions),
            llm_calls=llm_calls,
            graph_reroutes=graph_reroutes,
            tools_executed=tools_executed,
            total_time_ns=time.perf_counter_ns() - task_start,
            demoted_goal=demoted_goal,
        )
    
    def _log_decision(self, action: str, path=None, cost=0.0, 
                      failed_tool=None, detail=""):
        """Log a routing decision for observability."""
        self._decisions.append(RoutingDecision(
            action=action,
            path=path,
            failed_tool=failed_tool,
            cost=cost,
            timestamp_ns=time.perf_counter_ns(),
            detail=detail,
        ))
    
    @staticmethod
    def _default_executor(tool_name: str, request: dict) -> ExecutionResult:
        """Default mock executor — always succeeds."""
        return ExecutionResult(tool_name=tool_name, success=True, output={})
    
    @staticmethod
    def _default_llm_handler(goal: str, request: dict, 
                             completed: list[str]) -> Optional[str]:
        """Default LLM handler — returns a demoted goal string."""
        return f"degraded_{goal}"
    
    def reset(self):
        """Reset orchestrator state."""
        self._decisions.clear()
        self.graph.reset()
        self.monitors.reset()
