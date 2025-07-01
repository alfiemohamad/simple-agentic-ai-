from neo4j import GraphDatabase
from typing import List
from app.config import settings
import json

class Neo4jMemory:
    def __init__(self, uri: str, user: str, password: str, k: int = 5):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.k = k

    def add(self, user_input: str, agent_output: str) -> None:
        # Pastikan input dan output berupa string
        if not isinstance(user_input, str):
            user_input = json.dumps(user_input, ensure_ascii=False)
        if not isinstance(agent_output, str):
            agent_output = json.dumps(agent_output, ensure_ascii=False)
        with self.driver.session() as session:
            session.run(
                "CREATE (m:Memory {user_input: $input, agent_output: $output, timestamp: datetime()})",
                input=user_input,
                output=agent_output,
            )

    def load(self) -> List[str]:
        with self.driver.session() as session:
            res = session.run(
                "MATCH (m:Memory) RETURN m.user_input AS input, m.agent_output AS output "
                "ORDER BY m.timestamp DESC LIMIT $k",
                k=self.k
            )
            return [f"User: {r['input']}\nAgent: {r['output']}" for r in res]


