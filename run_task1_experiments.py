from task1_human_in_loop import app, AgentState

# =========================================================
# EXPERIMENT (a): Ambiguous Query
# =========================================================
print("\n=======================================================")
print("RUNNING EXPERIMENT (a): AMBIGUOUS QUERY")
print("=======================================================")

config_a = {"configurable": {"thread_id": "thread_ambiguous"}}

initial_state_a: AgentState = {
    "dialogue": [{
        "role": "user", 
        "content": "Evaluate the Silverbox RMSE model, but I didn't specify whether to use the 3072 train subset or the full 4096 dataset slice."
    }],
    "decision_count": 0,
    "next_role": "orchestrator",
    "instruction": "",
    "human_question": None
}

# 1. Run until graph interrupts at 'human'
for event in app.stream(initial_state_a, config=config_a):
    pass

state_a = app.get_state(config_a)

if state_a.next:
    print(f"\n>>> Graph Paused as Expected. Question: {state_a.values.get('human_question')}")

    # 2. Provide Human Reply
    human_reply_a = "Follow the 4096-sample slice (42650:46746) used in Homework 1 Task 5."
    print(f"\n👤 [HUMAN REPLIED]: {human_reply_a}")

    current_dialogue = state_a.values.get("dialogue", [])
    current_dialogue.append({"role": "human", "content": human_reply_a})

    # Update dialogue state cleanly without forcing next_role
    app.update_state(config_a, {"dialogue": current_dialogue}, as_node="human")

    # 3. Resume graph execution
    for event in app.stream(None, config=config_a):
        pass


# =========================================================
# EXPERIMENT (b): Hard Decision (Refuse once, then Accept)
# =========================================================
print("\n\n=======================================================")
print("RUNNING EXPERIMENT (b): HARD DECISION (REFUSE THEN ACCEPT)")
print("=======================================================")

config_b = {"configurable": {"thread_id": "thread_hard_decision"}}

initial_state_b: AgentState = {
    "dialogue": [{
        "role": "user", 
        "content": "Execute the PINN code tool immediately and overwrite test_plot.png on disk."
    }],
    "decision_count": 0,
    "next_role": "orchestrator",
    "instruction": "",
    "human_question": None
}

# 1. Run until first pause (Hard Decision prompt)
for event in app.stream(initial_state_b, config=config_b):
    pass

state_b1 = app.get_state(config_b)

if state_b1.next:
    print(f"\n>>> First Pause. Question: {state_b1.values.get('human_question')}")

    # 2. Refuse once
    refusal_msg = "No, do not run the code tool or overwrite disk right now. Review parameters first."
    print(f"\n👤 [HUMAN REPLIED]: {refusal_msg}")

    dialogue_b = state_b1.values.get("dialogue", [])
    dialogue_b.append({"role": "human", "content": refusal_msg})

    app.update_state(config_b, {"dialogue": dialogue_b}, as_node="human")

    # 3. Resume -> Orchestrator processes refusal
    for event in app.stream(None, config=config_b):
        pass

    state_b2 = app.get_state(config_b)
    if state_b2.next:
        print(f"\n>>> Second Pause. Question: {state_b2.values.get('human_question')}")
        
        # 4. Accept
        accept_msg = "Yes, parameters are verified. You may proceed and run the code tool."
        print(f"\n👤 [HUMAN REPLIED]: {accept_msg}")
        
        dialogue_b2 = state_b2.values.get("dialogue", [])
        dialogue_b2.append({"role": "human", "content": accept_msg})
        app.update_state(config_b, {"dialogue": dialogue_b2}, as_node="human")
        
        for event in app.stream(None, config=config_b):
            pass

print("\n✅ Task 1 Experiments Executed Successfully!")