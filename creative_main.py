# This keeps my BabyAGI experiment sandboxed from all the other work
from agent import CreativeAgent

if __name__ == "__main__":
    objective = "Compose a dark neo-soul piece inspired by existential literature and late-night jazz"

    agent = CreativeAgent(objective=objective)
    agent.run()