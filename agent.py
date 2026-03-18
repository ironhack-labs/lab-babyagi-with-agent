# Full Multi-Agent Version
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from planner import TaskPlanner
from memory import VectorMemory
from tools import get_tools
from rich import print

# Load environment variables FIRST
load_dotenv()


class CreativeAgent:
    def __init__(self, objective: str):
        self.objective = objective

        # 🔥 Separate models (important)
        self.planner_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

        self.composer_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.85
        )

        self.critic_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.6
        )

        # Components
        self.memory = VectorMemory()
        self.planner = TaskPlanner(self.planner_llm)
        self.tools = get_tools(self.composer_llm)

        self.task_list = []

    def run(self):
        print(f"[bold cyan]Objective:[/bold cyan] {self.objective}")

        # Initial tasks
        self.task_list = self.planner.create_initial_tasks(self.objective)

        iteration = 0

        while self.task_list and iteration < 6:
            iteration += 1

            print(f"\n[bold yellow]Iteration {iteration}[/bold yellow]")

            task = self.task_list.pop(0)
            print(f"[green]Executing Task:[/green] {task}")

            # 🔹 Composer Agent
            result = self.execute_task(task)
            print(f"[blue]Result:[/blue] {result[:200]}...\n")

            # 🔹 Critic Agent
            critique = self.critique(result)
            print(f"[red]Critique:[/red] {critique}\n")

            # 🔹 Store BOTH (critical for learning)
            combined = f"Task: {task}\nResult:\n{result}\n\nCritique:\n{critique}"
            self.memory.add(task, combined)

            # 🔹 Planner generates smarter tasks (now aware of critique)
            new_tasks = self.planner.generate_new_tasks(
                objective=self.objective,
                last_result=combined,
                pending_tasks=self.task_list
            )

            # 🔴 Filter junk / duplicates
            new_tasks = [
                t for t in new_tasks
                if len(t) > 10 and t not in self.task_list
            ]

            print(f"[magenta]New Tasks:[/magenta] {new_tasks}")

            self.task_list.extend(new_tasks)

    # -------------------------
    # Composer Agent
    # -------------------------
    def execute_task(self, task: str) -> str:
        context = self.memory.query(task)

        prompt = f"""
You are a composer AI specializing in music and literature.

Objective:
{self.objective}

Relevant past work:
{context}

Current task:
{task}

Produce a refined, creative, high-quality result.
"""

        response = self.composer_llm.invoke(prompt)
        return response.content

    # -------------------------
    # Critic Agent
    # -------------------------
    def critique(self, output: str) -> str:
        prompt = f"""
You are a strict creative critic.

Evaluate the work based on:
- originality
- emotional depth
- stylistic coherence
- alignment with the objective

Objective:
{self.objective}

Work:
{output}

Provide:
1. A concise critique
2. Two specific improvements
"""

        response = self.critic_llm.invoke(prompt)
        return response.content