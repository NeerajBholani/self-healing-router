"""
Parallel Health Monitors with priority-based competition.

Lightweight modules that run in parallel on every request, each producing
a priority-scored signal. The highest-priority signal wins via max().

Reference: Section 2.1 of the paper.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalType(Enum):
    INTENT = "intent"
    RISK = "risk"
    TOOL_HEALTH = "tool_health"
    MEMORY = "memory"
    PROGRESS = "progress"


@dataclass
class MonitorSignal:
    """A priority-scored signal from a health monitor."""
    source: SignalType
    priority: float  # 0.0 to 1.0
    payload: dict
    
    def __repr__(self):
        return f"Signal({self.source.value}, priority={self.priority:.2f})"


class IntentClassifier:
    """Regex/keyword-based intent classification. No LLM needed."""
    
    def __init__(self):
        self.patterns = {
            'refund': ['refund', 'return', 'money back', 'cancel order'],
            'booking': ['book', 'reserve', 'flight', 'hotel', 'travel'],
            'moderation': ['review', 'moderate', 'flag', 'report', 'content'],
            'support': ['help', 'issue', 'problem', 'assist'],
        }
    
    def evaluate(self, request: dict) -> MonitorSignal:
        text = request.get('text', '').lower()
        matched_intent = 'unknown'
        
        for intent, keywords in self.patterns.items():
            if any(kw in text for kw in keywords):
                matched_intent = intent
                break
        
        # Intent signals typically have moderate priority
        return MonitorSignal(
            source=SignalType.INTENT,
            priority=0.90 if matched_intent != 'unknown' else 0.30,
            payload={'intent': matched_intent, 'text': text}
        )


class RiskDetector:
    """Threshold-based risk detection."""
    
    def __init__(self, high_value_threshold: float = 10000.0):
        self.high_value_threshold = high_value_threshold
    
    def evaluate(self, request: dict) -> MonitorSignal:
        amount = request.get('amount', 0.0)
        risk_flags = request.get('risk_flags', [])
        
        # High-value transactions get high priority
        if amount > self.high_value_threshold:
            return MonitorSignal(
                source=SignalType.RISK,
                priority=0.95,
                payload={'risk_level': 'high', 'amount': amount, 'reason': 'high_value'}
            )
        
        if risk_flags:
            return MonitorSignal(
                source=SignalType.RISK,
                priority=0.85,
                payload={'risk_level': 'medium', 'flags': risk_flags}
            )
        
        return MonitorSignal(
            source=SignalType.RISK,
            priority=0.20,
            payload={'risk_level': 'low'}
        )


class ToolHealthMonitor:
    """
    Ping-based / circuit breaker tool health monitoring.
    Fires with highest priority (0.99) when a tool is down.
    """
    
    def __init__(self):
        self.tool_status: dict[str, bool] = {}  # tool_name -> healthy
        self.failure_counts: dict[str, int] = {}
        self.circuit_breaker_threshold: int = 3
    
    def report_failure(self, tool_name: str):
        """Record a tool failure."""
        self.failure_counts[tool_name] = self.failure_counts.get(tool_name, 0) + 1
        if self.failure_counts[tool_name] >= self.circuit_breaker_threshold:
            self.tool_status[tool_name] = False
    
    def report_success(self, tool_name: str):
        """Record a tool success."""
        self.tool_status[tool_name] = True
        self.failure_counts[tool_name] = 0
    
    def mark_down(self, tool_name: str):
        """Immediately mark a tool as down."""
        self.tool_status[tool_name] = False
    
    def mark_up(self, tool_name: str):
        """Mark a tool as healthy."""
        self.tool_status[tool_name] = True
        self.failure_counts[tool_name] = 0
    
    def get_failed_tools(self) -> list[str]:
        """Return list of tools currently marked as down."""
        return [name for name, healthy in self.tool_status.items() if not healthy]
    
    def evaluate(self, request: dict) -> MonitorSignal:
        failed_tools = [name for name, healthy in self.tool_status.items() if not healthy]
        
        if failed_tools:
            return MonitorSignal(
                source=SignalType.TOOL_HEALTH,
                priority=0.99,  # Highest priority — tool failures demand immediate attention
                payload={'failed_tools': failed_tools, 'status': 'degraded'}
            )
        
        return MonitorSignal(
            source=SignalType.TOOL_HEALTH,
            priority=0.10,
            payload={'failed_tools': [], 'status': 'healthy'}
        )


class ProgressTracker:
    """Tracks step completion for task integrity verification."""
    
    def __init__(self):
        self.completed_steps: list[str] = []
        self.expected_steps: list[str] = []
    
    def set_expected(self, steps: list[str]):
        self.expected_steps = steps
        self.completed_steps = []
    
    def mark_complete(self, step: str):
        self.completed_steps.append(step)
    
    def evaluate(self, request: dict) -> MonitorSignal:
        if not self.expected_steps:
            return MonitorSignal(
                source=SignalType.PROGRESS,
                priority=0.10,
                payload={'progress': 0.0}
            )
        
        progress = len(self.completed_steps) / len(self.expected_steps)
        return MonitorSignal(
            source=SignalType.PROGRESS,
            priority=0.50,
            payload={
                'progress': progress,
                'completed': list(self.completed_steps),
                'remaining': [s for s in self.expected_steps if s not in self.completed_steps]
            }
        )


class MonitorBank:
    """
    Runs all monitors in parallel and returns the winning signal.
    
    The winner is determined by max(priority) — a simple comparison
    that costs microseconds instead of an LLM call.
    
    Reference: Section 2.1, priority competition.
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.risk_detector = RiskDetector()
        self.tool_health = ToolHealthMonitor()
        self.progress_tracker = ProgressTracker()
    
    def run_all(self, request: dict) -> tuple[MonitorSignal, list[MonitorSignal]]:
        """
        Run all monitors and return (winner, all_signals).
        Winner is the signal with highest priority.
        """
        signals = [
            self.intent_classifier.evaluate(request),
            self.risk_detector.evaluate(request),
            self.tool_health.evaluate(request),
            self.progress_tracker.evaluate(request),
        ]
        
        winner = max(signals, key=lambda s: s.priority)
        return winner, signals
    
    def reset(self):
        """Reset all monitor state."""
        self.tool_health.tool_status.clear()
        self.tool_health.failure_counts.clear()
        self.progress_tracker.completed_steps.clear()
        self.progress_tracker.expected_steps.clear()
