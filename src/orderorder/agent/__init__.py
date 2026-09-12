"""The agent layer: the engine's checks, exposed to a Strands agent.

Three modules, and the split is the argument. `tools.py` wraps checks that already existed and adds no
judgement of its own; `model.py` decides which model answers and says so; `assistant.py` holds the
prompt that stops the model answering from memory. The engine underneath is unchanged -- the
verification graph still runs its eight nodes in a fixed order, and the agent chooses which question
to ask, never what the answer is.

    from orderorder.agent import build_assistant

    assistant = build_assistant()
    print(assistant.model.describe())
    print(assistant.ask("Is (2019) 4 SCC 1 still good law?"))
"""

from orderorder.agent.assistant import SYSTEM_PROMPT, Assistant, build_assistant
from orderorder.agent.model import ModelChoice, NoModelConfigured, choose_model
from orderorder.agent.tools import build_tools

__all__ = [
    "SYSTEM_PROMPT",
    "Assistant",
    "ModelChoice",
    "NoModelConfigured",
    "build_assistant",
    "build_tools",
    "choose_model",
]
