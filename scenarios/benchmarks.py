"""
19 Benchmark Scenarios from the paper evaluation.

Three domains:
- Customer Support (7 scenarios): Linear pipeline topology
- Travel Booking (6 scenarios): Dependency DAG topology  
- Content Moderation (6 scenarios): Parallel fan-out topology

Each scenario defines:
- A tool graph with nodes and edges
- Which tools fail
- Expected behavior (reroute vs escalate)
- Whether the task should complete successfully

Reference: Section 4 and Tables 2-5 of the paper.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Scenario:
    """A benchmark scenario for evaluation."""
    id: str
    name: str
    domain: str
    topology: str  # "linear", "dag", "fanout"
    description: str
    
    # Graph definition
    nodes: list[str]
    edges: list[tuple[str, str, float]]  # (source, target, cost)
    start: str
    goal: str
    
    # Failure injection
    failed_tools: list[str]
    
    # Expected outcomes
    expected_success: bool
    expected_reroutes: int
    expected_llm_calls: int
    expected_path: Optional[list[str]] = None


def get_all_scenarios() -> list[Scenario]:
    """Return all 19 benchmark scenarios."""
    return (
        customer_support_scenarios() + 
        travel_booking_scenarios() + 
        content_moderation_scenarios()
    )


def customer_support_scenarios() -> list[Scenario]:
    """
    Customer Support domain — Linear pipeline topology.
    
    Base graph:
    Start → CRM → Stripe/Razorpay → Email/SMS → Complete
    
    7 scenarios testing single failures, compound failures,
    and all-tools-down escalation.
    """
    base_nodes = ['Start', 'CRM', 'Stripe', 'Razorpay', 'Email', 'SMS', 'Complete']
    base_edges = [
        ('Start', 'CRM', 0.1),
        ('CRM', 'Stripe', 0.3),
        ('CRM', 'Razorpay', 0.5),
        ('Stripe', 'Email', 0.3),
        ('Stripe', 'SMS', 0.4),
        ('Razorpay', 'Email', 0.3),
        ('Razorpay', 'SMS', 0.4),
        ('Email', 'Complete', 0.1),
        ('SMS', 'Complete', 0.1),
    ]
    
    return [
        Scenario(
            id="CS-1", name="Happy path",
            domain="customer_support", topology="linear",
            description="All tools healthy. Optimal path: Start→CRM→Stripe→Email→Complete",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=[],
            expected_success=True, expected_reroutes=0, expected_llm_calls=0,
            expected_path=["Start", "CRM", "Stripe", "Email", "Complete"],
        ),
        Scenario(
            id="CS-2", name="Single payment failure",
            domain="customer_support", topology="linear",
            description="Stripe is down. Should reroute to Razorpay.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["Stripe"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=0,
            expected_path=["Start", "CRM", "Razorpay", "Email", "Complete"],
        ),
        Scenario(
            id="CS-3", name="Single notification failure",
            domain="customer_support", topology="linear",
            description="Email is down. Should reroute to SMS.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["Email"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=0,
        ),
        Scenario(
            id="CS-4", name="Compound: payment + notification",
            domain="customer_support", topology="linear",
            description="Stripe AND Email both down. Should reroute through Razorpay→SMS.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["Stripe", "Email"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=0,
            expected_path=["Start", "CRM", "Razorpay", "SMS", "Complete"],
        ),
        Scenario(
            id="CS-5", name="All payments down",
            domain="customer_support", topology="linear",
            description="Both Stripe and Razorpay down. No payment path — LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["Stripe", "Razorpay"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=1,
        ),
        Scenario(
            id="CS-6", name="All notifications down",
            domain="customer_support", topology="linear",
            description="Both Email and SMS down. Payment succeeds but no notification path.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["Email", "SMS"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=1,
        ),
        Scenario(
            id="CS-7", name="Cascading: CRM down",
            domain="customer_support", topology="linear",
            description="CRM is the entry point and it's down. No path at all — immediate LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["CRM"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=1,
        ),
    ]


def travel_booking_scenarios() -> list[Scenario]:
    """
    Travel Booking domain — Dependency DAG topology.
    
    Base graph:
    Start → FlightAPI/TrainAPI → HotelAPI/HostelAPI → CarAPI/BusAPI → Payment → Confirm
    
    Dependencies: Hotel depends on Flight (need dates), Car depends on Hotel (need location).
    6 scenarios testing DAG-aware rerouting.
    """
    base_nodes = ['Start', 'FlightAPI', 'TrainAPI', 'HotelAPI', 'HostelAPI', 
                  'CarAPI', 'BusAPI', 'Payment', 'Confirm']
    base_edges = [
        ('Start', 'FlightAPI', 0.3),
        ('Start', 'TrainAPI', 0.5),
        ('FlightAPI', 'HotelAPI', 0.3),
        ('FlightAPI', 'HostelAPI', 0.4),
        ('TrainAPI', 'HotelAPI', 0.3),
        ('TrainAPI', 'HostelAPI', 0.4),
        ('HotelAPI', 'CarAPI', 0.3),
        ('HotelAPI', 'BusAPI', 0.4),
        ('HostelAPI', 'CarAPI', 0.4),
        ('HostelAPI', 'BusAPI', 0.3),
        ('CarAPI', 'Payment', 0.2),
        ('BusAPI', 'Payment', 0.2),
        ('Payment', 'Confirm', 0.1),
    ]
    
    return [
        Scenario(
            id="TB-1", name="Happy path",
            domain="travel_booking", topology="dag",
            description="All tools healthy. Optimal: Start→Flight→Hotel→Car→Payment→Confirm",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=[],
            expected_success=True, expected_reroutes=0, expected_llm_calls=0,
        ),
        Scenario(
            id="TB-2", name="Flight API down",
            domain="travel_booking", topology="dag",
            description="FlightAPI down. Reroute to TrainAPI.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=["FlightAPI"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=0,
        ),
        Scenario(
            id="TB-3", name="Hotel + Car both down",
            domain="travel_booking", topology="dag",
            description="Compound failure: HotelAPI and CarAPI. Reroute to Hostel→Bus.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=["HotelAPI", "CarAPI"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=0,
        ),
        Scenario(
            id="TB-4", name="All transport down",
            domain="travel_booking", topology="dag",
            description="FlightAPI and TrainAPI both down. No transport — LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=["FlightAPI", "TrainAPI"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=1,
        ),
        Scenario(
            id="TB-5", name="Triple compound failure",
            domain="travel_booking", topology="dag",
            description="Flight, Hotel, and Car all down. Reroute: Train→Hostel→Bus.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=["FlightAPI", "HotelAPI", "CarAPI"],
            expected_success=True, expected_reroutes=3, expected_llm_calls=0,
        ),
        Scenario(
            id="TB-6", name="Payment system down",
            domain="travel_booking", topology="dag",
            description="Single point of failure: Payment is the only payment node. LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Confirm",
            failed_tools=["Payment"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=1,
        ),
    ]


def content_moderation_scenarios() -> list[Scenario]:
    """
    Content Moderation domain — Parallel fan-out topology.
    
    Base graph:
    Start → TextMod/ImageMod/VideoMod → ToxicityScorer → HumanReview/AutoApprove → Complete
    
    Fan-out: Multiple moderation tools can run in parallel.
    6 scenarios testing partial completion and graceful degradation.
    """
    base_nodes = ['Start', 'TextMod', 'ImageMod', 'VideoMod',
                  'ToxicityScorer', 'HumanReview', 'AutoApprove', 'Complete']
    base_edges = [
        ('Start', 'TextMod', 0.2),
        ('Start', 'ImageMod', 0.3),
        ('Start', 'VideoMod', 0.5),
        ('TextMod', 'ToxicityScorer', 0.2),
        ('ImageMod', 'ToxicityScorer', 0.3),
        ('VideoMod', 'ToxicityScorer', 0.4),
        ('ToxicityScorer', 'HumanReview', 0.5),
        ('ToxicityScorer', 'AutoApprove', 0.2),
        ('HumanReview', 'Complete', 0.1),
        ('AutoApprove', 'Complete', 0.1),
    ]
    
    return [
        Scenario(
            id="CM-1", name="Happy path",
            domain="content_moderation", topology="fanout",
            description="All tools healthy. Cheapest: Start→TextMod→Toxicity→AutoApprove→Complete",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=[],
            expected_success=True, expected_reroutes=0, expected_llm_calls=0,
        ),
        Scenario(
            id="CM-2", name="Text moderator down",
            domain="content_moderation", topology="fanout",
            description="TextMod down. Reroute to ImageMod path.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["TextMod"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=0,
        ),
        Scenario(
            id="CM-3", name="Auto-approve down",
            domain="content_moderation", topology="fanout",
            description="AutoApprove down. Route to HumanReview instead.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["AutoApprove"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=0,
        ),
        Scenario(
            id="CM-4", name="Compound: text + auto-approve",
            domain="content_moderation", topology="fanout",
            description="TextMod and AutoApprove both down. Route: ImageMod→Toxicity→HumanReview.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["TextMod", "AutoApprove"],
            expected_success=True, expected_reroutes=2, expected_llm_calls=0,
        ),
        Scenario(
            id="CM-5", name="All moderators down",
            domain="content_moderation", topology="fanout",
            description="TextMod, ImageMod, VideoMod all down. No moderation path — LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["TextMod", "ImageMod", "VideoMod"],
            expected_success=True, expected_reroutes=3, expected_llm_calls=1,
        ),
        Scenario(
            id="CM-6", name="Toxicity scorer down",
            domain="content_moderation", topology="fanout",
            description="Single point of failure: ToxicityScorer. No path to review — LLM escalation.",
            nodes=list(base_nodes), edges=list(base_edges),
            start="Start", goal="Complete",
            failed_tools=["ToxicityScorer"],
            expected_success=True, expected_reroutes=1, expected_llm_calls=1,
        ),
    ]
