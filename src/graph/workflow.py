from __future__ import annotations

from langfuse import get_client
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
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="span",
            name="router-node",
            input={"business_requirement": state.business_requirement},
        ) as span:
            routing_result = self.router_agent.route(state.business_requirement)
            output = {"detected_services": routing_result.detected_services}
            span.update(output=output)
            return output

    def run(self, state: PolicyState):
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="span",
            name="policy-workflow-run",
            input={
                "business_requirement": state.business_requirement,
                "graph_structure": list(self.app.get_graph().nodes.keys()),
            },
        ) as span:
            result = self.app.invoke(state)
            detected_services = (
                result["detected_services"]
                if isinstance(result, dict)
                else result.detected_services
            )
            span.update(output={"detected_services": detected_services})
            return result
