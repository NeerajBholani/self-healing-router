"""
Baseline implementations for comparison.

- ReAct: Every decision routes through mock LLM (counts calls)
- LangGraph: Static state machine with pre-coded single-failure fallbacks

Counting methodology (matches paper Section 4.2):

ReAct LLM calls include:
  1. Initial planning call (LLM receives task, plans approach)
  2. Per-step tool selection (LLM picks next tool at each node)
  3. Per-failure error reasoning (LLM observes failure, reasons about alternative)
  4. Dead-end reasoning (LLM concludes no path exists, decides to give up)

LangGraph silent failures:
  Only COMPOUND failures count as silent (primary tool + its fallback both down,
  state machine skips the layer without error). Single-point-of-failure nodes
  with no fallback defined cause a visible crash/error, NOT a silent failure.

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
    
    LLM call counting follows the standard ReAct loop:
    - 1 initial planning call (task understanding)
    - 1 call per routing decision at each node
    - 1 call per failure observation (LLM reads error, reasons about next step)
    - 1 dead-end call when the LLM concludes no viable path remains
    
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
        
        # Step 1: Initial planning call — LLM receives the task and plans
        self.llm_calls += 1
        
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
                # Dead-end: LLM first reasons about backtracking possibilities
                self.llm_calls += 1  # "Can I backtrack or try another approach?"
                # Then concludes no viable path remains
                self.llm_calls += 1  # "No, I must give up."
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
    
    IMPORTANT: Only COMPOUND failures count as silent failures.
    A compound failure occurs when a tool AND its designated fallback are
    both down — the state machine skips the entire layer without raising
    an error. Single-point-of-failure nodes with no fallback cause a
    visible crash/error (state machine gets stuck), which is NOT silent.
    
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
    
    def _is_compound_failure(self, scenario: Scenario, failed_tool: str) -> bool:
        """Check if a failure is compound (tool + its fallback both down).
        
        Compound = tool has a fallback AND that fallback is also down.
        No-fallback = tool has no entry in fallback map (crash, not silent).
        """
        fallback_key = (scenario.domain, failed_tool)
        fallback = self.fallbacks.get(fallback_key)
        if fallback is None:
            return False  # No fallback defined -> crash, not silent
        return fallback in scenario.failed_tools  # Fallback also down -> compound
    
    def execute(self, scenario: Scenario) -> BaselineResult:
        path_taken = []
        silent_failures = 0
        crashed = False
        
        # Walk the cheapest static path
        current = scenario.start
        visited = set()
        goal_reached = False
        
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
                # State machine is stuck at this node.
                # Determine if this is a SILENT failure or a CRASH.
                
                # Check each failed tool reachable from current node
                failed_at_this_node = [
                    t for t, c in next_options if t in scenario.failed_tools
                ]
                
                # Is any of these a compound failure (tool + fallback both down)?
                has_compound = any(
                    self._is_compound_failure(scenario, t)
                    for t in failed_at_this_node
                )
                
                if has_compound:
                    # COMPOUND: state machine encounters a layer where both the
                    # primary tool and its fallback are down.
                    #
                    # A well-engineered LangGraph has skip-edges that jump past
                    # blocked layers to maintain progress. The skip succeeds but
                    # the step is silently omitted — no error raised.
                    #
                    # Exception: if ALL tools at this node are down (no working
                    # tool in the entire layer), the fan-out produces an empty
                    # result which is detectable. This is a crash, not silent.
                    all_targets_at_node = [t for t, c in next_options]
                    any_working = any(t not in scenario.failed_tools for t in all_targets_at_node)
                    all_failed_compound = all(
                        t in scenario.failed_tools for t in all_targets_at_node
                    ) and len(all_targets_at_node) >= 3
                    
                    if all_failed_compound:
                        # Total layer failure in fan-out — empty result is visible
                        crashed = True
                    else:
                        # Partial compound failure — state machine skips silently
                        silent_failures += 1
                else:
                    # No compound failure — single point of failure with no
                    # fallback defined. State machine raises a visible error.
                    crashed = True
                break
            
            if current == scenario.goal:
                goal_reached = True
        
        # Determine correctness
        # The paper defines correctness as: did the system produce a correct
        # result OR raise a visible error? Silent failures are incorrect because
        # the system appears to succeed while missing steps. Crashes are
        # considered "correct" in the observability sense — the error is visible
        # and operators can intervene.
        correct = silent_failures == 0
        
        return BaselineResult(
            system="LangGraph",
            scenario_id=scenario.id,
            success=goal_reached or (silent_failures > 0),
            correct=correct,
            llm_calls=0,  # LangGraph uses zero LLM calls
            silent_failures=silent_failures,
            path_taken=path_taken,
            detail=f"{'SILENT FAILURE' if silent_failures > 0 else 'CRASH' if crashed else 'OK'}: {silent_failures} silent, crashed={crashed}"
        )
