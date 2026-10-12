import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from simulated_bench import OpticalBenchSimulator

# 1. Setup API Key & LLM
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")

llm = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0.1
)

# 2. Global Bench Instance & Results Storage
bench = OpticalBenchSimulator(dmd_res=32, cam_res=64)
experiment_results = []

def run_bench_tool(macropixel_size: int, num_iterations: int) -> str:
    res = bench.run_focusing_search(macropixel_size=macropixel_size, num_iterations=num_iterations)
    exp_num = len(experiment_results) + 1
    
    # Save camera images
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(res["I_initial"], cmap='inferno')
    axes[0].set_title(f"Initial Speckle (Enh: {res['initial_enhancement']:.2f})")
    axes[1].imshow(res["I_final"], cmap='inferno')
    axes[1].set_title(f"Focused Spot (Enh: {res['final_enhancement']:.2f})")
    
    img_filename = f"focus_experiment_{exp_num}.png"
    plt.savefig(img_filename, dpi=200)
    plt.close()

    summary = {
        "experiment": exp_num,
        "macropixel_size": macropixel_size,
        "iterations": num_iterations,
        "initial_enhancement": round(res["initial_enhancement"], 2),
        "final_enhancement": round(res["final_enhancement"], 2),
        "image": img_filename
    }
    experiment_results.append(summary)
    
    return f"Experiment {exp_num} Completed! Initial Enh: {summary['initial_enhancement']}, Final Enh: {summary['final_enhancement']}. Image saved to {img_filename}."

# 3. State Definition
class AgentState(TypedDict):
    dialogue: List[dict]
    decision_count: int
    next_role: str
    instruction: str

# 4. Orchestrator Node
def orchestrator_node(state: AgentState):
    decision_count = state.get("decision_count", 0)
    dialogue = state.get("dialogue", [])

    if len(experiment_results) >= 3 or decision_count >= 6:
        return {"next_role": "finish", "instruction": "Task completed with full experiment table.", "decision_count": decision_count + 1}

    prompt = (
        "You are the Optical Bench Orchestrator.\n"
        "Your goal is to optimize beam focusing using DMD pattern search.\n"
        "Choose 'scientist' to decide parameters, or 'executor' to call the focusing tool.\n"
        "Respond ONLY with raw JSON:\n"
        '{"next_role": "scientist"|"executor"|"finish", "instruction": "short step description"}'
    )
    
    try:
        resp = llm.invoke(f"{prompt}\nRecent dialogue: {dialogue[-2:]}")
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        data = json.loads(content)
        next_role = data.get("next_role", "executor")
        instruction = data.get("instruction", "Run bench experiment")
    except Exception:
        next_role = "executor"
        instruction = "Proceed with tool call."

    print(f"\n🎯 [Orchestrator Decision #{decision_count + 1}] -> Next Role: {next_role}")
    return {"next_role": next_role, "instruction": instruction, "decision_count": decision_count + 1, "dialogue": dialogue + [{"role": "orchestrator", "content": f"Route to {next_role}"}]}

# 5. Scientist Node
def scientist_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    configs = [(2, 250), (4, 300), (8, 200)]
    curr_exp = len(experiment_results)
    macro, iters = configs[curr_exp % len(configs)]
    msg = f"Scientist: Designed experiment configuration - Macropixel Size: {macro}, Iterations: {iters}"
    print(f"\n🔬 [Scientist]: {msg}")
    return {"dialogue": dialogue + [{"role": "scientist", "content": msg}]}

# 6. Executor Node
def executor_node(state: AgentState):
    dialogue = state.get("dialogue", [])
    configs = [(2, 250), (4, 300), (8, 200)]
    curr_exp = len(experiment_results)
    macro, iters = configs[curr_exp % len(configs)]
    
    print(f"\n⚙️ [Executor]: Invoking Simulated Bench Tool with Macropixel Size={macro}, Iterations={iters}...")
    result_msg = run_bench_tool(macropixel_size=macro, num_iterations=iters)
    print(f"⚙️ [Executor Result]: {result_msg}")
    
    return {"dialogue": dialogue + [{"role": "executor", "content": result_msg}]}

# 7. Workflow Assembly
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("scientist", scientist_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    lambda state: state["next_role"],
    {"scientist": "scientist", "executor": "executor", "finish": END}
)

workflow.add_edge("scientist", "orchestrator")
workflow.add_edge("executor", "orchestrator")

app = workflow.compile()

# 8. Main Execution Loop
if __name__ == "__main__":
    print("🚀 Starting Agentic Beam Focusing Task 2...")
    initial_state = {
        "dialogue": [{"role": "user", "content": "Autonomous beam focusing optimization."}],
        "decision_count": 0,
        "next_role": "orchestrator",
        "instruction": ""
    }
    app.invoke(initial_state)

    print("\n=======================================================")
    print("📊 EXPERIMENT RESULTS TABLE (FOR TASK 2 REPORT)")
    print("=======================================================")
    print(f"{'Exp #':<6} | {'Macropixel Size':<15} | {'Iterations':<10} | {'Initial Enh':<12} | {'Final Enh':<10}")
    print("-" * 65)
    for row in experiment_results:
        print(f"{row['experiment']:<6} | {row['macropixel_size']:<15} | {row['iterations']:<10} | {row['initial_enhancement']:<12} | {row['final_enhancement']:<10}")
    print("=======================================================")