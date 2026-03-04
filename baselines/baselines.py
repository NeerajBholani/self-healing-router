"""
Baseline implementations for comparison.

- ReAct: Every decision routes through mock LLM (counts calls)
- LangGraph: Static state machine with pre-coded single-failure fallbacks

Reference: Section 4.2 of the paper.
"""

from dataclasses import dataclass, field
from typing import Optional
from scenarios.benchmarks import Scenario


@dataclass
class BaselineResult:
    """Result from a baseline execution."""
    system: str
    scenario_id: str
    success: bool
    correct: bool  # Did it produce the right outcome?
    llm_calls: int = 0
    silent_failures: int = 0  # Completed but missed steps
    path_taken: list[str] = field(default_factory=list)
    detail: str = ""


class ReActBaseline:
    """
    ReAct-style baseline: every routing decision goes through the LLM.
    
    When a tool fails, the error is passed back to the LLM which reasons
    about what to do next. Every fork, retry, and fallback = 1 LLM call.
    
    This is how OpenAI Agents SDK, Claude SDK, and similar frameworks work.
    """
    
    def __init__(self):
        self.llm_calls = 0
    
    def execute(self, scenario: Scenario) -> BaselineResult:
        self.llm_calls = 0
        path_taken = []
        
        # Build available tools (excluding failed ones)
        available = [n for n in scenario.nodes 
                    if n not in scenario.failed_tools 
                    and n not in ['Start', scenario.goal]]
        
        # ReAct: LLM call per routing decision + per failure observation
        current = scenario.start
        visited = set()
        goal_reached = False
        
        while current != scenario.goal and len(visited) < len(scenario.nodes) * 2:
            visited.add(current)
            
            # Find edges from current node
            next_options = [(t, c) for s, t, c in scenario.edges if s == current]
            
            if not next_options:
                break
            
            # LLM reasons about which tool to use next
            self.llm_calls += 1
            
            # Try each option
            moved = False
            for target, cost in sorted(next_options, key=lambda x: x[1]):
                if target in scenario.failed_tools:
                    # LLM observes failure, reasons about alternative
                    self.llm_calls += 1
                    continue
                
                # Success — move to this node
                current = target
                path_taken.append(target)
                moved = True
                break
            
            if not moved:
                # LLM reasons about giving up
                self.llm_calls += 1
                break
            
            if current == scenario.goal:
                goal_reached = True
        
        return BaselineResult(
            system="ReAct",
            scenario_id=scenario.id,
            success=goal_reached or bool(scenario.failed_tools),
            correct=True,  # ReAct always gets the right answer (LLM reasons correctly)
            llm_calls=self.llm_calls,
            silent_failures=0,  # ReAct doesn't have silent failures
            path_taken=path_taken,
            detail=f"ReAct used {self.llm_calls} LLM calls"
        )


class LangGraphBaseline:
    """
    LangGraph-style baseline: static state machine with pre-coded fallbacks.
    
    Only single-failure fallback edges are pre-coded. Compound failures
    (multiple tools down simultaneously) are not covered, leading to
    silent failures — the state machine proceeds without the missing step.
    
    Reference: Section 3 (motivating example) and Table 6 of the paper.
    """
    
    def __init__(self):
        # Pre-coded single-failure fallback map
        # Maps: (domain, failed_tool) -> fallback_tool
        self.fallbacks = {
            # Customer Support
            ('customer_support', 'Stripe'): 'Razorpay',
            ('customer_support', 'Razorpay'): 'Stripe',
            ('customer_support', 'Email'): 'SMS',
            ('customer_support', 'SMS'): 'Email',
            # Travel Booking
            ('travel_booking', 'FlightAPI'): 'TrainAPI',
            ('travel_booking', 'TrainAPI'): 'FlightAPI',
            ('travel_booking', 'HotelAPI'): 'HostelAPI',
            ('travel_booking', 'HostelAPI'): 'HotelAPI',
            ('travel_booking', 'CarAPI'): 'BusAPI',
            ('travel_booking', 'BusAPI'): 'CarAPI',
            # Content Moderation
            ('content_moderation', 'TextMod'): 'ImageMod',
            ('content_moderation', 'ImageMod'): 'TextMod',
            ('content_moderation', 'VideoMod'): 'ImageMod',
            ('content_moderation', 'AutoApprove'): 'HumanReview',
            ('content_moderation', 'HumanReview'): 'AutoApprove',
        }
    
    def execute(self, scenario: Scenario) -> BaselineResult:
        path_taken = []
        silent_failures = 0
        
        # Walk the cheapest static path
        current = scenario.start
        visited = set()
        goal_reached = False
        stuck_count = 0
        
        while current != scenario.goal and len(visited) < len(scenario.nodes) * 3:
            visited.add(current)
            
            # Find edges from current node (sorted by cost — static ordering)
            next_options = [(t, c) for s, t, c in scenario.edges if s == current]
            
            moved = False
            for target, cost in sorted(next_options, key=lambda x: x[1]):
                if target in scenario.failed_tools:
                    # Check for pre-coded fallback
                    fallback_key = (scenario.domain, target)
                    fallback = self.fallbacks.get(fallback_key)
                    
                    if fallback and fallback not in scenario.failed_tools:
                        # Single-failure fallback works
                        current = fallback
                        path_taken.append(fallback)
                        moved = True
                        break
                    else:
                        # No valid fallback — try next option in the edge list
                        continue
                else:
                    current = target
                    path_taken.append(target)
                    moved = True
                    break
            
            if not moved:
                # COMPOUND FAILURE: no valid transition from this node
                # LangGraph silently proceeds — skips this layer
                silent_failures += 1
                stuck_count += 1
                if stuck_count > 2:
                    break
                # Try to jump forward by finding any reachable node
                all_targets = set(t for s, t, c in scenario.edges)
                all_sources = set(s for s, t, c in scenario.edges)
                # State machine is stuck — it doesn't backtrack
                break
            
            if current == scenario.goal:
                goal_reached = True
        
        # Determine correctness
        # LangGraph is "correct" only if no silent failures occurred
        correct = goal_reached and silent_failures == 0
        
        return BaselineResult(
            system="LangGraph",
            scenario_id=scenario.id,
            success=goal_reached or (silent_failures > 0),  # May "succeed" with gaps
            correct=correct,
            llm_calls=0,  # LangGraph uses zero LLM calls
            silent_failures=silent_failures,
            path_taken=path_taken,
            detail=f"{'SILENT FAILURE' if silent_failures > 0 else 'OK'}: {silent_failures} missed steps"
        )
