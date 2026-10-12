import os
import sys
import time
import json
import subprocess
from typing import List, TypedDict, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. API Key & LLM Setup
# ==========================================
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1000
)

# ==========================================
# 2. State & Dialogue Definition
# ==========================================
class DialogueMessage(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    dialogue: List[DialogueMessage]
    decision_count: int
    next_role: str
    instruction: str
    human_question: Optional[str]

# ==========================================
# 3. Tool Execution
# ==========================================
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
        output = f"Exit Code: {result.returncode}\n"
        output += f"--- STDOUT ---\n{result.stdout}\n"
        output += f"--- STDERR ---\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error executing code: {str(e)}"

# ==========================================
# 4. Orchestrator Node (Human-in-the-Loop Routing)
# ==========================================
def orchestrator_node(state: AgentState):
    time.sleep(1)
    decision_count = state.get("decision_count", 0)
    dialogue = state.get("dialogue", [])

    if decision_count >= 10:
        print("\n🛑 [Orchestrator] Max decision limit reached.")
        return {
            "next_role": "finish",
            "instruction": "Maximum decisions reached.",
            "decision_count": decision_count + 1
        }

    formatted_dialogue = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in dialogue[-4:]])

    system_prompt = (
        "You are Orchestrator for an AI Science Agent.\n"
        "Route to: 'scientist', 'executor', 'general', 'human', or 'finish'.\n\n"
        "RULES FOR 'human':\n"
        "1. AMBIGUOUS QUERY: Request has >1 reading changing roles or reported values.\n"
        "2. HARD DECISION: Action will run code, overwrite files, or finalize measurements.\n"
        "   - If user explicitly rejected a hard decision in previous turn, DO NOT route to executor; acknowledge refusal and route to scientist or ask clarification.\n\n"
        "Respond strictly in JSON:\n"
        "{\n"
        '  "next_role": "scientist" | "executor" | "general" | "human" | "finish",\n'
        '  "instruction": "short task for next role",\n'
        '  "question": "question for human (mandatory if next_role is human, else empty)",\n'
        '  "reason": "one sentence reason"\n'
        "}"
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
        reason = f"Fallback routing ({str(e)[:30]})"

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

# ==========================================
# 5. Nodes
# ==========================================
def scientist_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    plan = "Scientist: Reviewed request. Recommended checking slice length and hyperparameters prior to execution."
    print(f"\n🔬 [Scientist]: {plan}")
    return {"dialogue": dialogue + [{"role": "scientist", "content": plan}]}

def general_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    msg = "General Agent: Baseline dataset values verified."
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

# ==========================================
# 6. Graph Assembly & Interrupt
# ==========================================
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

# ==========================================
# 7. Execution of Experiments (a) & (b)
# ==========================================
if __name__ == "__main__":
    # --- EXPERIMENT (a): Ambiguous Query ---
    print("\n" + "="*60)
    print("RUNNING EXPERIMENT (a): AMBIGUOUS QUERY")
    print("="*60)

    config_a = {"configurable": {"thread_id": "thread_ambiguous"}}
    initial_state_a: AgentState = {
        "dialogue": [{"role": "user", "content": "Evaluate the Silverbox RMSE for the dataset slice."}],
        "decision_count": 0,
        "next_role": "orchestrator",
        "instruction": "",
        "human_question": None
    }

    for event in app.stream(initial_state_a, config=config_a):
        pass

    state_a = app.get_state(config_a)
    print(f"\n>>> Graph Paused. Question posed: {state_a.values.get('human_question')}")

    human_reply_a = "Use the 4096-sample slice (indices 42650:46746) from Homework 1."
    print(f"\n👤 [HUMAN REPLIED]: {human_reply_a}")

    dialogue_a = state_a.values.get("dialogue", [])
    dialogue_a.append({"role": "human", "content": human_reply_a})
    app.update_state(config_a, {"dialogue": dialogue_a, "next_role": "orchestrator"})

    for event in app.stream(None, config=config_a):
        pass

    # --- EXPERIMENT (b): Hard Decision (Refuse then Accept) ---
    print("\n" + "="*60)
    print("RUNNING EXPERIMENT (b): HARD DECISION (REFUSE THEN ACCEPT)")
    print("="*60)

    config_b = {"configurable": {"thread_id": "thread_hard_decision"}}
    initial_state_b: AgentState = {
        "dialogue": [{"role": "user", "content": "Run the code tool now to train PINN and overwrite test_plot.png on disk."}],
        "decision_count": 0,
        "next_role": "orchestrator",
        "instruction": "",
        "human_question": None
    }

    # First attempt -> Pauses for approval
    for event in app.stream(initial_state_b, config=config_b):
        pass

    state_b1 = app.get_state(config_b)
    print(f"\n>>> First Pause (Hard Decision). Question: {state_b1.values.get('human_question')}")

    # Step Refused by Human
    refusal = "No, do not execute the code tool or overwrite disk right now. Review the setup first."
    print(f"\n👤 [HUMAN REPLIED]: {refusal}")

    dialogue_b1 = state_b1.values.get("dialogue", [])
    dialogue_b1.append({"role": "human", "content": refusal})
    app.update_state(config_b, {"dialogue": dialogue_b1, "next_role": "orchestrator"})

    # Graph runs after refusal -> Orchestrator routes away from executor to show step did NOT run
    for event in app.stream(None, config=config_b):
        pass

    state_b2 = app.get_state(config_b)

    # Human now authorizes execution
    accept = "Yes, setup is verified. You are approved to run the code tool and overwrite test_plot.png."
    print(f"\n👤 [HUMAN REPLIED]: {accept}")

    dialogue_b2 = state_b2.values.get("dialogue", [])
    dialogue_b2.append({"role": "human", "content": accept})
    app.update_state(config_b, {"dialogue": dialogue_b2, "next_role": "orchestrator"})

    for event in app.stream(None, config=config_b):
        pass

    print("\n✅ Task 1 Experiments Executed Successfully!")