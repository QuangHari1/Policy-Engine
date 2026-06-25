from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.router_agent import RouterAgent
from src.graph.state import PolicyState


class PolicyWorkflow:
    def __init__(self, router_agent: RouterAgent):
        self.router_agent = router_agent
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PolicyState)
        graph.add_node("router", self._router_node)
        graph.add_edge(START, "router")
        graph.add_edge("router", END)
        return graph.compile()

    def _router_node(self, state: PolicyState):
        routing_result = self.router_agent.route(state.business_requirement)
        return {"detected_services": routing_result.detected_services}

    def run(self, state: PolicyState):
        return self.app.invoke(state)
