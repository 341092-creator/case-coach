import json
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import coach_core as cc

CASE = ("Mokobara is a travel luggage brand. " * 60)
BRIEF = {"is_business_case": True, "injection_detected": False, "title": "Mokobara", "company": "Mokobara",
         "industry": "Luggage", "decision_problem": "Whether to open offline stores", "key_facts": ["fact"]}
QS = {"questions": [{"id": 1, "question": "What is the core problem?", "hint": "see para 1", "framework_link": "Strengths"}]}
FB = {"scores": {"grounding": 5, "framework": 3, "insight": 9, "actionability": 2, "clarity": 4},
      "reasoning": {k: "r" for k, _ in cc.RUBRIC}, "strengths": ["s"], "gaps": ["g"],
      "followup_question": "Why?", "off_topic_or_injection": False, "refusal": ""}
SUM = {"overall_comment": "ok", "top_strengths": ["a"], "top_priorities_to_improve": ["b"], "next_steps": ["c"]}
calls = []
def fake(client, model, prompt, **kw):
    calls.append(prompt[:30])
    if "Read the case" in prompt: return BRIEF
    if "Socratic coaching questions" in prompt: return QS
    if "Assess the student" in prompt: return FB
    return SUM

with patch("coach_core.call_gemini", fake), patch("google.genai.Client", lambda **k: object()):
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert at.warning, "should ask for key"
    at.sidebar.text_input[0].set_value("fake").run()
    # too-short case -> error, no API call
    at.text_area[0].set_value("tiny case").run()
    at.button[[b.label for b in at.button].index("🚀 Analyse case & generate coaching questions")].click().run()
    assert any("only 2 words" in e.value for e in at.error) and not calls
    # valid case
    at.text_area[0].set_value(CASE).run()
    at.button[[b.label for b in at.button].index("🚀 Analyse case & generate coaching questions")].click().run()
    assert not at.exception and at.session_state["questions"], [e.value for e in at.error]
    # short answer rejected
    ta = [t for t in at.text_area if t.key == "ans_1"][0]; ta.set_value("short").run()
    [b for b in at.button if b.key == "fb_1"][0].click().run()
    assert any("only 1 words" in e.value for e in at.error)
    # real answer -> feedback, score clamped (9 -> 5)
    ans = "Mokobara has a strong online brand but offline expansion costs rise quickly. " * 4
    [t for t in at.text_area if t.key == "ans_1"][0].set_value(ans).run()
    [b for b in at.button if b.key == "fb_1"][0].click().run()
    assert at.session_state["feedback"][1]["scores"]["insight"] == 5
    n = len(calls)
    # double submit -> no new API call
    [b for b in at.button if b.key == "fb_1"][0].click().run()
    assert len(calls) == n, "duplicate submit hit API"
    # summary
    [b for b in at.button if b.label == "Generate final summary"][0].click().run()
    assert at.session_state["summary"]["overall_comment"] == "ok" and not at.exception
print("UI TESTS PASSED; api calls:", len(calls))
