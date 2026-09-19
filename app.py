"""Streamlit UI: WhatsApp Scam Triage & Privacy Advisor (student prototype)."""
import json
import os

import pandas as pd
import streamlit as st

# Read the API key from Streamlit secrets (Cloud) into the environment; env vars also work locally.
try:
    for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file locally
    pass

import agent  # noqa: E402  (after env is set)
import rag  # noqa: E402

st.set_page_config(page_title="WhatsApp Scam Triage Agent", page_icon="🛡️", layout="wide")

MAX_RUNS_PER_SESSION = 20  # protects the free API quota on a public demo
SAMPLES = {
    "OTP takeover attempt": "Hey sorry, I typed your number by mistake and my WhatsApp code came to you. Please read it out to me quickly, I'm locked out!",
    "Fake delivery link": "Your package is on hold. Confirm your address and pay a small $2 fee here: post-parcel-track.example.info/pay",
    "Fake job (task scam)": "Earn 500 per day from home! Just rate 20 products online. Join our channel, pay 1000 refundable deposit to start: jobs-fast.example.xyz",
    "Boss gift-card request": "Hi, it's Mark from head office, new number. I'm stuck in a board meeting. Please grab 3 gift cards for a client and send me the photos of the codes.",
    "Ambiguous stranger": "Hello, is this Priya? Nice to meet you at the conference. Can we connect here?",
    "Legitimate message": "Hi! Your table for 4 at 8pm on Saturday is confirmed. See you then. - Lotus Cafe",
}
BADGE = {"escalate": "🔴 ESCALATE", "report": "🟠 REPORT", "warn": "🟡 WARN", "no_action": "🟢 NO ACTION"}


@st.cache_resource(show_spinner="Loading embedding model and building the vector index (first run takes a minute)...")
def warm_up():
    rag.get_store()
    return rag.backend_name()


backend = warm_up()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🛡️ Scam Triage Agent")
    st.caption("Student capstone prototype. Not affiliated with WhatsApp or Meta.")
    if agent.llm_available():
        st.success("Gemini API key found")
    else:
        st.warning("No API key: running in rule-based fallback mode")
    st.caption(f"Embeddings: {backend}")
    st.caption(f"Knowledge base: {len(rag.load_scams())} examples, {len(rag.load_kb())} policy/tip entries")
    st.divider()
    st.info(
        "Use only synthetic or sample messages. Emails and phone numbers are masked before the LLM call, "
        "but on the free Gemini tier Google may use prompts to improve its products."
    )
    st.session_state.setdefault("runs", 0)
    st.caption(f"Analyses this session: {st.session_state['runs']}/{MAX_RUNS_PER_SESSION}")


def render_result(res: dict, show_trace: bool = True):
    cls, dec, rep = res["classification"], res["decision"], res["report"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label", cls["label"].title())
    c2.metric("Scam type", cls["scam_type"].replace("_", " "))
    c3.metric("Confidence", f"{cls['confidence']:.0%}")
    c4.metric("Priority", f"P{dec['priority']}")
    st.subheader(BADGE[dec["action"]])
    for r in dec["reasons"]:
        st.write(f"- {r}")
    st.caption(f"Policy applied: {rag.get_policy(dec['policy_id'])['title']}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Red flags")
        for f in cls["red_flags"] or ["None found"]:
            st.write(f"- {f}")
        st.markdown("#### Most similar known cases (RAG)")
        st.dataframe(
            pd.DataFrame([{"similarity": s["similarity"], "category": s["category"], "example": s["message"][:90]}
                          for s in res["similar_scams"]]),
            hide_index=True,
        )
    with right:
        st.markdown("#### 3 privacy-settings tips")
        for i, t in enumerate(res["privacy_tips"], 1):
            with st.expander(f"{i}. {t['title']}", expanded=(i == 1)):
                st.write(t["how_to"])
                st.caption(f"Why: {t['why']}")

    st.markdown("#### Final report")
    st.info(f"**For the reviewer:** {rep['reviewer_summary']}")
    st.success(f"**For the user:** {rep['user_message']}")
    for s in rep["next_steps"]:
        st.write(f"- {s}")
    if not res["llm_used"]:
        st.warning("LLM was unavailable for classification, so rule-based fallbacks were used.")

    if show_trace:
        with st.expander("Agent trace (tool calls)"):
            st.json(res["trace"])
    st.download_button("Download case JSON", json.dumps(res, indent=2, ensure_ascii=False),
                       file_name=f"case_{res['case_id']}.json", mime="application/json", key=f"dl_{res['case_id']}")


tab1, tab2, tab3 = st.tabs(["Triage one report", "Review queue (at scale)", "How it works"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.title("WhatsApp Scam Triage & Privacy Advisor")
    st.write("Paste a reported message. The agent classifies it, finds similar known scams, decides what to do and gives privacy-settings advice.")
    choice = st.selectbox("Load a sample", ["(type your own)"] + list(SAMPLES))
    default = SAMPLES.get(choice, "")
    msg = st.text_area("Reported message", value=default, height=130, key=f"msg_{choice}")
    a, b = st.columns(2)
    times = a.number_input("Times this message was reported by other users", 0, 500, 0)
    fwd = b.checkbox("It arrived as a forward or in a group chat")

    if st.button("Analyze", type="primary", disabled=not msg.strip()):
        if st.session_state["runs"] >= MAX_RUNS_PER_SESSION:
            st.error("Session limit reached to protect the free API quota. Reload the page to continue.")
        else:
            st.session_state["runs"] += 1
            with st.status("Agent running...", expanded=True) as status:
                res = agent.run_agent(
                    msg, {"times_reported": int(times), "forwarded": fwd},
                    on_step=lambda s: st.write(f"✅ `{s['tool']}` ({s['seconds']}s)"),
                )
                status.update(label="Done", state="complete", expanded=False)
            st.session_state["last"] = res
    if "last" in st.session_state:
        render_result(st.session_state["last"])

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Why an agent? Reviewers can't read every report")
    st.write("Simulate a small queue of reports. The agent triages each one and sorts the queue so humans see the riskiest cases first.")
    n = st.slider("Reports in the queue", 2, 6, 4)
    if st.button("Triage the queue"):
        items = list(SAMPLES.items())[:n]
        if st.session_state["runs"] + n > MAX_RUNS_PER_SESSION:
            st.error("Not enough session quota left for this batch.")
        else:
            rows, bar = [], st.progress(0.0)
            for i, (name, text) in enumerate(items, 1):
                r = agent.run_agent(text)
                st.session_state["runs"] += 1
                rows.append({"priority": r["decision"]["priority"], "action": r["decision"]["action"],
                             "type": r["classification"]["scam_type"], "confidence": r["classification"]["confidence"],
                             "report": name, "why": "; ".join(r["decision"]["reasons"])[:110]})
                bar.progress(i / len(items))
            df = pd.DataFrame(rows).sort_values("priority")
            st.dataframe(df, hide_index=True)
            st.caption(f"{(df['action'] == 'escalate').sum()} escalated to humans, {(df['action'] == 'no_action').sum()} closed automatically.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        """
**Pipeline (plain Python tool functions)**
1. `extract_signals`: regex red flags (OTP request, APK, urgency, links...) and PII masking
2. `classify_message`: Gemini labels scam / phishing / suspicious / legitimate
3. `retrieve_similar_scams`: ChromaDB + local sentence-transformers finds the closest known cases
4. *Self-check:* if confidence is low or retrieval disagrees, the agent re-classifies using the retrieved evidence
5. `decide_action`: transparent rules map the result to **warn / report / escalate** (policies P1-P4)
6. `get_privacy_tips`: retrieves and picks exactly 3 privacy-settings tips
7. `generate_report`: reviewer summary + plain-language message for the user

**Limits:** synthetic data (50 examples), English-only, policies are illustrative, and the LLM can be wrong; escalated cases are meant for a human to review.
"""
    )
