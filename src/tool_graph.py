"""
Tool Graph with Dijkstra-based shortest-path routing.

Nodes represent tools, edges represent capabilities with cost weights.
When a tool fails, its edges are reweighted to infinity and Dijkstra
finds the next cheapest working path automatically.

Reference: Section 2.2 of the paper.
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional


INF = float('inf')


@dataclass
class Edge:
    """A directed edge in the tool graph."""
    source: str
    target: str
    base_cost: float
    current_cost: float = 0.0
    
    def __post_init__(self):
        if self.current_cost == 0.0:
            self.current_cost = self.base_cost


@dataclass
class ToolNode:
    """A tool in the graph."""
    name: str
    healthy: bool = True
    latency_ms: float = 100.0
    reliability: float = 0.99
    rate_limit_proximity: float = 0.0  # 0.0 = far from limit, 1.0 = at limit
    metadata: dict = field(default_factory=dict)


class ToolGraph:
    """
    Cost-weighted directed graph of tools with Dijkstra routing.
    
    Supports dynamic edge reweighting based on health telemetry,
    enabling automatic rerouting when tools fail.
    """
    
    def __init__(self):
        self.nodes: dict[str, ToolNode] = {}
        self.edges: list[Edge] = []
        self._adjacency: dict[str, list[Edge]] = {}
    
    def add_node(self, name: str, **kwargs) -> ToolNode:
        """Add a tool node to the graph."""
        node = ToolNode(name=name, **kwargs)
        self.nodes[name] = node
        if name not in self._adjacency:
            self._adjacency[name] = []
        return node
    
    def add_edge(self, source: str, target: str, cost: float) -> Edge:
        """Add a directed edge between tools."""
        edge = Edge(source=source, target=target, base_cost=cost)
        self.edges.append(edge)
        self._adjacency.setdefault(source, []).append(edge)
        return edge
    
    def get_edges_for_node(self, node_name: str) -> list[Edge]:
        """Get all edges connected to a node (both directions)."""
        connected = []
        for edge in self.edges:
            if edge.source == node_name or edge.target == node_name:
                connected.append(edge)
        return connected
    
    def reweight_failed_node(self, node_name: str):
        """
        Set all edges touching a failed node to infinity.
        This is O(degree) where degree is typically 2-4.
        Reference: Section 2.3, Step 2.
        """
        self.nodes[node_name].healthy = False
        for edge in self.get_edges_for_node(node_name):
            edge.current_cost = INF
    
    def restore_node(self, node_name: str):
        """Restore a node's edges to base cost (tool recovered)."""
        self.nodes[node_name].healthy = True
        for edge in self.get_edges_for_node(node_name):
            edge.current_cost = edge.base_cost
    
    def compute_weight(self, edge: Edge) -> float:
        """
        Multi-factor weight composition.
        W(tool) = base_cost × latency(t) × reliability(t) × rate_limit(t)
        Reference: Section 2.5, Equation 1.
        """
        if edge.current_cost == INF:
            return INF
        
        target_node = self.nodes.get(edge.target)
        if not target_node or not target_node.healthy:
            return INF
        
        # Normalize latency factor (100ms baseline)
        latency_factor = max(0.5, target_node.latency_ms / 100.0)
        # Reliability factor (inverse — lower reliability = higher cost)
        reliability_factor = 1.0 / max(0.01, target_node.reliability)
        # Rate limit proximity (approaching limit = higher cost)
        rate_limit_factor = 1.0 + (target_node.rate_limit_proximity * 5.0)
        
        return edge.base_cost * latency_factor * reliability_factor * rate_limit_factor
    
    def dijkstra(self, start: str, goal: str) -> Optional[tuple[list[str], float]]:
        """
        Dijkstra's shortest-path algorithm on the tool graph.
        
        Returns (path, total_cost) or None if no path exists.
        O((V + E) log V) with binary heap.
        
        Reference: Section 2.3, Step 3.
        """
        if start not in self.nodes or goal not in self.nodes:
            return None
        
        start_time = time.perf_counter_ns()
        
        # Distance and predecessor tracking
        dist = {name: INF for name in self.nodes}
        prev = {name: None for name in self.nodes}
        dist[start] = 0.0
        
        # Min-heap: (distance, node_name)
        heap = [(0.0, start)]
        visited = set()
        
        while heap:
            d, u = heapq.heappop(heap)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == goal:
                break
            
            if d > dist[u]:
                continue
            
            for edge in self._adjacency.get(u, []):
                v = edge.target
                if v in visited:
                    continue
                
                weight = self.compute_weight(edge)
                new_dist = dist[u] + weight
                
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))
        
        elapsed_ns = time.perf_counter_ns() - start_time
        
        # Reconstruct path
        if dist[goal] == INF:
            return None
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[goal], elapsed_ns
    
    def reset(self):
        """Reset all edges to base costs and all nodes to healthy."""
        for node in self.nodes.values():
            node.healthy = True
            node.latency_ms = 100.0
            node.reliability = 0.99
            node.rate_limit_proximity = 0.0
        for edge in self.edges:
            edge.current_cost = edge.base_cost
    
    def __repr__(self):
        healthy = sum(1 for n in self.nodes.values() if n.healthy)
        return (f"ToolGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"healthy={healthy}/{len(self.nodes)})")
