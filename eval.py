"""Quick evaluation on held-out messages (not in the vector index). Run: python eval.py"""
import json
import time
from pathlib import Path

import agent

cases = json.loads((Path(__file__).parent / "data" / "test_messages.json").read_text(encoding="utf-8"))
ok_label = ok_action = 0
print(f"{'expected':<12}{'got_label':<12}{'action':<10}{'exp_action':<16}message")
for c in cases:
    r = agent.run_agent(c["message"])
    label, action = r["classification"]["label"], r["decision"]["action"]
    pred_bad = label in ("scam", "phishing", "suspicious")
    ok_label += pred_bad == c["malicious"]
    exp = c["action"]
    ok_action += (action in ("warn", "report")) if exp == "warn_or_report" else (action == exp)
    print(f"{('malicious' if c['malicious'] else 'legit'):<12}{label:<12}{action:<10}{exp:<16}{c['message'][:45]}")
    time.sleep(4 if agent.llm_available() else 0)  # stay under free-tier rate limits
n = len(cases)
print(f"\nDetection accuracy: {ok_label}/{n} = {ok_label/n:.0%}   Action accuracy: {ok_action}/{n} = {ok_action/n:.0%}")
