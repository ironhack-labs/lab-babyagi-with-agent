class TaskPlanner:
    def __init__(self, llm):
        self.llm = llm

    def create_initial_tasks(self, objective: str):
        prompt = f"""
Break this creative objective into 3–5 clear tasks.

Objective:
{objective}

Return each task on a new line.
"""

        response = self.llm.invoke(prompt)

        tasks = [
            t.strip("- ").strip()
            for t in response.content.split("\n")
            if t.strip()
        ]

        return tasks

    def generate_new_tasks(self, objective, last_result, pending_tasks):
        prompt = f"""
Objective:
{objective}

Last result:
{last_result}

Pending tasks:
{pending_tasks}

Suggest up to 2 new refinement tasks.
Return each on a new line.
"""

        response = self.llm.invoke(prompt)

        tasks = [
            t.strip("- ").strip()
            for t in response.content.split("\n")
            if t.strip()
        ]

        return tasks