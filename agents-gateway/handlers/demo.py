from agents import Agent

from registry import register

agent = Agent(
    name="DemoAgent",
    model="gpt-5-mini",
    instructions=(
        "You are DemoAgent v1, running inside the agents-gateway service. "
        "Always start your response with the exact phrase: '[DemoAgent✓]' "
        "so the user can confirm this agent is being called correctly. "
        "Then answer the user's question helpfully."
    ),
)

register("demo_agent", agent)
