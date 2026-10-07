import hashlib
import json
import os

import streamlit as st

import coach_core as cc

st.set_page_config(page_title="MBA Case-Study Coach", page_icon="🎓", layout="wide")

DEFAULT_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3-flash-preview"]


# ------------------------------------------------------------------ state helpers
def init_state():
    defaults = {
        "case_text": "", "brief": None, "framework": "SWOT", "level": "With hints",
        "questions": [], "answers": {}, "feedback": {}, "summary": None,
        "busy": False, "warnings": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def export_session() -> str:
    keys = ["case_text", "brief", "framework", "level", "questions", "answers", "feedback", "summary"]
    return json.dumps({k: st.session_state.get(k) for k in keys}, indent=1)


def import_session(raw: bytes):
    data = json.loads(raw.decode("utf-8"))
    for k in ["case_text", "brief", "framework", "level", "questions", "answers", "feedback", "summary"]:
        if k in data:
            st.session_state[k] = data[k]
    # JSON turns int keys to strings; normalise
    st.session_state["answers"] = {int(k): v for k, v in st.session_state["answers"].items()}
    st.session_state["feedback"] = {int(k): v for k, v in st.session_state["feedback"].items()}


def get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def digest(*parts) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


init_state()

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 Case Coach")
    st.caption("AI coach for MBA case analysis. You do the thinking; the coach asks, probes, and scores against a rubric.")

    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = secret_key or st.text_input("Gemini API key", type="password",
                                          help="Free key from aistudio.google.com. Not stored.")
    model = st.selectbox("Model", DEFAULT_MODELS, index=0,
                         help="If one model is retired or rate-limited, try another.")

    st.info("**Privacy:** the case text and your answers are sent to Google's Gemini API. "
            "On the free tier Google may use inputs to improve its products. "
            "Do not paste confidential or personal data.", icon="🔒")
    st.warning("AI feedback is **formative, not an official grade**, and can be wrong. "
               "Always cross-check against the case.", icon="⚠️")

    st.divider()
    st.subheader("Save / resume session")
    st.download_button("⬇️ Download session", export_session(), "case_coach_session.json",
                       "application/json", use_container_width=True)
    up = st.file_uploader("Resume a saved session", type=["json"], key="resume")
    if up is not None and st.button("Load session", use_container_width=True):
        try:
            import_session(up.getvalue())
            st.success("Session restored.")
            st.rerun()
        except Exception:
            st.error("That file is not a valid session export.")

    if st.button("🗑️ Start over", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k != "resume":
                del st.session_state[k]
        st.rerun()

st.title("MBA Case-Study Coach")

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = get_client(api_key)

# ------------------------------------------------------------- step 1: load case
st.header("1 · Load the case")
tab_paste, tab_upload = st.tabs(["Paste text", "Upload file"])
raw_text = ""
with tab_paste:
    pasted = st.text_area("Paste the case text", value=st.session_state["case_text"], height=220,
                          placeholder="Paste at least 250 words of the case...")
    raw_text = pasted
with tab_upload:
    f = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt", "md"])
    if f is not None:
        try:
            raw_text = cc.extract_text(f.name, f.getvalue())
            st.success(f"Read {len(raw_text.split()):,} words from {f.name}.")
        except ValueError as e:
            st.error(str(e))

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    framework = st.selectbox("Framework to coach on", list(cc.FRAMEWORKS),
                             index=list(cc.FRAMEWORKS).index(st.session_state["framework"]))
    st.caption(cc.FRAMEWORKS[framework])
with c2:
    level = st.radio("Coaching level", ["With hints", "Socratic only"], horizontal=True,
                     index=["With hints", "Socratic only"].index(st.session_state["level"]))
with c3:
    n_q = st.slider("Questions", 3, 6, 4)

if st.button("🚀 Analyse case & generate coaching questions", type="primary",
             disabled=st.session_state["busy"]):
    text, warns, err = cc.validate_case(raw_text)
    if err:
        st.error(err)
    else:
        for w in warns:
            st.warning(w)
        st.session_state["busy"] = True
        try:
            with st.spinner("Reading the case..."):
                brief = cc.call_gemini(client, model, cc.prompt_case_brief(text),
                                       required_keys=("is_business_case", "decision_problem"))
            if brief.get("injection_detected"):
                st.warning("The text contains instructions aimed at the AI. They were ignored; "
                           "only the business content is used.")
            if not brief.get("is_business_case"):
                st.error("This does not look like a business case. "
                         + (brief.get("refusal") or "Please provide a case study."))
            else:
                with st.spinner("Preparing Socratic questions..."):
                    qs = cc.call_gemini(client, model,
                                        cc.prompt_questions(text, framework, level, n_q),
                                        temperature=0.4, required_keys=("questions",))
                if not qs["questions"]:
                    raise cc.CoachError("The coach did not return any questions. Please retry.")
                st.session_state.update(case_text=text, brief=brief, framework=framework,
                                        level=level, questions=qs["questions"], answers={},
                                        feedback={}, summary=None)
        except cc.CoachError as e:
            st.error(str(e))
        finally:
            st.session_state["busy"] = False

brief = st.session_state["brief"]
if brief:
    with st.expander(f"📄 Case brief: {brief.get('title', 'Case')}", expanded=True):
        st.markdown(f"**Company:** {brief.get('company', '-')}  |  **Industry:** {brief.get('industry', '-')}")
        st.markdown(f"**Core decision/problem:** {brief['decision_problem']}")
        if brief.get("key_facts"):
            st.markdown("**Key facts pulled from the case:**")
            for kf in brief["key_facts"]:
                st.markdown(f"- {kf}")

# ------------------------------------------------------- step 2: answer questions
if st.session_state["questions"]:
    st.header(f"2 · Work through the questions ({st.session_state['framework']})")
    for q in st.session_state["questions"]:
        qid = int(q["id"])
        with st.container(border=True):
            st.markdown(f"**Q{qid}. {q['question']}**")
            if q.get("framework_link"):
                st.caption(f"Tests: {q['framework_link']}")
            if q.get("hint") and st.session_state["level"] == "With hints":
                with st.expander("💡 Hint"):
                    st.write(q["hint"])
            ans = st.text_area("Your answer", value=st.session_state["answers"].get(qid, ""),
                               key=f"ans_{qid}", height=140)
            st.session_state["answers"][qid] = ans

            if st.button("Get feedback", key=f"fb_{qid}", disabled=st.session_state["busy"]):
                err = cc.validate_answer(ans)
                if err:
                    st.error(err)
                else:
                    sig = digest(st.session_state["case_text"][:2000], q["question"], ans)
                    existing = st.session_state["feedback"].get(qid)
                    if existing and existing.get("sig") == sig:
                        st.info("Same answer already assessed; showing saved feedback "
                                "(no duplicate API call).")
                    else:
                        st.session_state["busy"] = True
                        try:
                            with st.spinner("Assessing against the rubric..."):
                                fb = cc.call_gemini(
                                    client, model,
                                    cc.prompt_feedback(st.session_state["case_text"],
                                                       st.session_state["framework"],
                                                       q["question"], ans),
                                    temperature=0.0, required_keys=("scores",))
                            if fb.get("off_topic_or_injection") or fb.get("refusal"):
                                st.warning("The coach could not assess this answer: "
                                           + (fb.get("refusal") or "it looked off-topic."))
                            else:
                                fb["scores"] = cc.clamp_scores(fb["scores"])
                                fb["check"] = cc.grounding_check(st.session_state["case_text"], ans)
                                fb["sig"] = sig
                                st.session_state["feedback"][qid] = fb
                        except cc.CoachError as e:
                            st.error(str(e))
                        finally:
                            st.session_state["busy"] = False

            fb = st.session_state["feedback"].get(qid)
            if fb:
                scores = fb["scores"]
                valid = [v for v in scores.values() if v]
                avg = sum(valid) / len(valid) if valid else 0
                st.markdown(f"**Rubric score: {avg:.1f} / 5**")
                cols = st.columns(len(cc.RUBRIC))
                for col, (k, desc) in zip(cols, cc.RUBRIC):
                    col.metric(k.title(), f"{scores.get(k) or '-'}/5", help=desc)
                with st.expander("Why these scores"):
                    for k, _ in cc.RUBRIC:
                        st.markdown(f"- **{k.title()}:** {fb.get('reasoning', {}).get(k, '-')}")
                chk = fb["check"]
                st.caption(f"Rule-based check (no AI): {chk['numbers_in_case']}/{chk['numbers_cited']} "
                           f"figures and {chk['entities_in_case']}/{chk['entities_cited']} named terms "
                           f"in your answer appear in the case. {chk['answer_words']} words.")
                warn = cc.compare_ai_vs_rule(scores.get("grounding"), chk)
                if warn:
                    st.warning("Check this: " + warn)
                if fb.get("strengths"):
                    st.success("**Strengths:** " + " ".join(fb["strengths"]))
                if fb.get("gaps"):
                    st.error("**Gaps:** " + " ".join(fb["gaps"]))
                if fb.get("followup_question"):
                    st.info("**Coach asks:** " + fb["followup_question"])

# ------------------------------------------------------------ step 3: summary
if st.session_state["feedback"]:
    st.header("3 · Session summary")
    if st.button("Generate final summary", disabled=st.session_state["busy"]):
        st.session_state["busy"] = True
        try:
            data = [{"question": q["question"],
                     "answer": st.session_state["answers"].get(int(q["id"]), ""),
                     "scores": st.session_state["feedback"][int(q["id"])]["scores"],
                     "gaps": st.session_state["feedback"][int(q["id"])].get("gaps", [])}
                    for q in st.session_state["questions"]
                    if int(q["id"]) in st.session_state["feedback"]]
            with st.spinner("Writing summary..."):
                st.session_state["summary"] = cc.call_gemini(
                    client, model,
                    cc.prompt_summary(brief.get("title", "Case"), st.session_state["framework"], data),
                    required_keys=("overall_comment",))
        except cc.CoachError as e:
            st.error(str(e))
        finally:
            st.session_state["busy"] = False

    s = st.session_state["summary"]
    if s:
        st.markdown(f"**Overall:** {s['overall_comment']}")
        st.markdown("**Strengths**")
        for x in s.get("top_strengths", []):
            st.markdown(f"- {x}")
        st.markdown("**Priorities to improve**")
        for x in s.get("top_priorities_to_improve", []):
            st.markdown(f"- {x}")
        st.markdown("**Next steps**")
        for x in s.get("next_steps", []):
            st.markdown(f"- {x}")

        md = [f"# Case Coach report: {brief.get('title', 'Case')}", f"Framework: {st.session_state['framework']}", ""]
        for q in st.session_state["questions"]:
            i = int(q["id"])
            md += [f"## Q{i}. {q['question']}", "", st.session_state["answers"].get(i, ""), ""]
            if i in st.session_state["feedback"]:
                md.append("Scores: " + ", ".join(f"{k}={v}" for k, v in st.session_state["feedback"][i]["scores"].items()))
                md.append("")
        md += ["## Summary", s["overall_comment"], "", "_AI-generated formative feedback, not an official grade._"]
        st.download_button("⬇️ Download report (.md)", "\n".join(md), "case_coach_report.md")
