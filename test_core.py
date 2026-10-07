import json, coach_core as cc

# --- validation edge cases
assert cc.validate_case("")[2]
assert "only 5 words" in cc.validate_case("one two three four five")[2]
long = "word " * 40000
t, w, e = cc.validate_case(long); assert e is None and len(t) == cc.MAX_CASE_CHARS and w
assert cc.validate_answer("too short") and cc.validate_answer("") and cc.validate_answer("a "*30) is None
try: cc.extract_text("x.exe", b"1"); raise SystemExit("fail")
except ValueError as ex: assert "Unsupported" in str(ex)
try: cc.extract_text("bad.pdf", b"not a pdf"); raise SystemExit("fail")
except ValueError as ex: assert "Could not read" in str(ex)
assert cc.extract_text("a.txt", b"hello") == "hello"

# --- JSON robustness
assert cc.parse_json('```json\n{"a":1}\n```') == {"a":1}
assert cc.parse_json('Sure! {"a": 2} hope that helps') == {"a":2}
for bad in ["", "no json here"]:
    try: cc.parse_json(bad); raise SystemExit("fail")
    except Exception: pass

# --- score clamping (hallucinated 7/5, strings, missing)
s = cc.clamp_scores({"grounding": 7, "framework": "3", "insight": -2, "clarity": "abc"})
assert s == {"grounding":5,"framework":3,"insight":1,"actionability":None,"clarity":None}, s

# --- grounding check + AI-vs-rule disagreement
case = "Mokobara sold 1,200 units in 2023 with 35% margin across Delhi and Mumbai stores."
good = cc.grounding_check(case, "Mokobara sold 1,200 units with 35% margin in Delhi.")
assert good["numbers_cited"] == good["numbers_in_case"] == 2, good
bad = cc.grounding_check(case, "The company should expand, selling 9,999 units at 80% margin.")
assert len(bad["numbers_not_in_case"]) == 2, bad
generic = cc.grounding_check(case, "The company has strong brand and should grow globally soon.")
assert generic["low_grounding"]
assert cc.compare_ai_vs_rule(5, generic) and cc.compare_ai_vs_rule(5, bad)
assert cc.compare_ai_vs_rule(3, good) is None

# --- retry / failure paths with fake client
class R:  # fake response
    def __init__(s,t): s.text=t
class C:
    def __init__(s, outs): s.outs=list(outs); s.models=s; s.calls=0
    def generate_content(s, **kw):
        s.calls+=1; o=s.outs.pop(0)
        if isinstance(o, Exception): raise o
        return R(o)
import time; time.sleep=lambda x: None
c = C(["garbage", '{"scores": {}}']); assert cc.call_gemini(c,"m","p",required_keys=("scores",)) == {"scores":{}} and c.calls==2
c = C(["x","y","z"]);
try: cc.call_gemini(c,"m","p"); raise SystemExit("fail")
except cc.CoachError as ex: assert "malformed" in str(ex)
c = C([Exception("429 RESOURCE_EXHAUSTED")]*3)
try: cc.call_gemini(c,"m","p"); raise SystemExit("fail")
except cc.CoachError as ex: assert "rate limit" in str(ex)
c = C([Exception("404 NOT_FOUND model")])
try: cc.call_gemini(c,"m","p"); raise SystemExit("fail")
except cc.CoachError as ex: assert "retired" in str(ex) and c.calls==1
c = C([Exception("403 PERMISSION_DENIED API key")])
try: cc.call_gemini(c,"m","p"); raise SystemExit("fail")
except cc.CoachError as ex: assert "key" in str(ex)
print("ALL CORE TESTS PASSED")
