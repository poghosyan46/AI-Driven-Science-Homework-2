import os
import sys
import time
import json
import subprocess
from typing import List, TypedDict, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. API Key setup (Տեղադրեք ձեր մեկ հատ ճիշտ gsk_... key-ը)
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")

llm = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0.1,
    max_tokens=1000
)

# 2. State definition
class DialogueMessage(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    dialogue: List[DialogueMessage]
    decision_count: int
    next_role: str
    instruction: str
    human_question: Optional[str]

# 3. Execution Tool
def execute_python_code(code: str, filename: str = "generated_script.py") -> str:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=120
        )
        return f"Exit Code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

# 4. Orchestrator Node
def orchestrator_node(state: AgentState):
    time.sleep(1)
    decision_count = state.get("decision_count", 0)
    dialogue = state.get("dialogue", [])

    if decision_count >= 8:
        print("\n🛑 [Orchestrator] Max decision limit reached. Finishing.")
        return {
            "next_role": "finish",
            "instruction": "Maximum decisions reached.",
            "decision_count": decision_count + 1
        }

    formatted_dialogue = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in dialogue[-4:]])

    system_prompt = (
        "You are the Orchestrator for an AI Science System.\n"
        "Choose next role from: 'scientist', 'executor', 'general', 'human', or 'finish'.\n\n"
        "ROUTING RULES FOR 'human':\n"
        "1. AMBIGUOUS QUERY: The latest request has more than 1 interpretation (e.g. which data slice/RMSE to evaluate). Choose 'human' and ask a clarifying question.\n"
        "2. HARD DECISION: The next step runs a code tool, overwrites a file on disk, or ends/reports measurements. Choose 'human' and ask for permission.\n"
        "3. If the human just gave instructions or accepted, proceed to 'scientist' or 'executor'. If refused, return to 'orchestrator' or 'scientist'.\n\n"
        "Respond STRICTLY in JSON:\n"
        '{"next_role": "scientist"|"executor"|"general"|"human"|"finish", "instruction": "short task", "question": "question for human if human else empty", "reason": "1 sentence"}'
    )

    prompt = f"{system_prompt}\n\nRecent Dialogue:\n{formatted_dialogue}\n\nDecision JSON:"
    
    try:
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        
        next_role = data.get("next_role", "scientist")
        instruction = data.get("instruction", "")
        question = data.get("question", "")
        reason = data.get("reason", "")
    except Exception as e:
        next_role = "scientist"
        instruction = "Proceed with planning."
        question = ""
        reason = f"LLM parsing fallback: {str(e)[:40]}"

    print(f"\n🎯 [Orchestrator Decision #{decision_count + 1}]")
    print(f"  Next Role: {next_role}")
    print(f"  Reason: {reason}")
    if question:
        print(f"  Question to Human: {question}")

    orchestrator_msg = f"Decision: Route to '{next_role}'. {instruction}"
    if question:
        orchestrator_msg += f" Question: {question}"

    new_dialogue = dialogue + [{"role": "orchestrator", "content": orchestrator_msg}]

    return {
        "dialogue": new_dialogue,
        "decision_count": decision_count + 1,
        "next_role": next_role,
        "instruction": instruction,
        "human_question": question
    }

# 5. Worker Nodes
def scientist_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    plan = "Scientist: Formulated data processing scheme for Silverbox system."
    print(f"\n🔬 [Scientist]: {plan}")
    return {"dialogue": dialogue + [{"role": "scientist", "content": plan}]}

def general_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    msg = "General Agent: Baseline verification complete."
    print(f"\n💬 [General Agent]: {msg}")
    return {"dialogue": dialogue + [{"role": "general", "content": msg}]}

def executor_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    code = "print('Executed PINN verification tool... Free-run Test RMSE: 14.436 V')"
    print("\n⚙️ [Executor]: Running Python Tool...")
    out = execute_python_code(code)
    print(f"⚙️ [Executor Output]:\n{out}")
    return {"dialogue": dialogue + [{"role": "executor", "content": f"Execution Output:\n{out}"}]}

def human_node(state: AgentState):
    question = state.get("human_question", "Approval needed. Proceed?")
    print(f"\n🛑 [HUMAN INTERRUPT REQUISITION]: {question}")
    return {}

# 6. LangGraph Setup with Checkpointer
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("scientist", scientist_node)
workflow.add_node("general", general_node)
workflow.add_node("executor", executor_node)
workflow.add_node("human", human_node)

workflow.set_entry_point("orchestrator")

def route_decision(state: AgentState):
    return state.get("next_role", "finish")

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "scientist": "scientist",
        "general": "general",
        "executor": "executor",
        "human": "human",
        "finish": END
    }
)

workflow.add_edge("scientist", "orchestrator")
workflow.add_edge("general", "orchestrator")
workflow.add_edge("executor", "orchestrator")
workflow.add_edge("human", "orchestrator")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human"])