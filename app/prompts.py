LETTERS = "ABCDE"

SYSTEM_PROMPT = (
    "You are ClinicMate, a medical question-answering assistant focused on healthcare "
    "in African settings. Answer accurately and concisely using established clinical "
    "knowledge and consider resource-limited settings and diseases common in Africa. "
    "You do not replace a health professional."
)

REFUSAL = (
    "I'm ClinicMate, and I can only help with health and medical questions. "
    "Please ask me something about symptoms, diseases, treatments or healthcare."
)

DISCLAIMER = (
    "\n\n---\n*ClinicMate is a student research project, not medical advice. "
    "In an emergency, go to the nearest health facility.*"
)

def truncate(text, max_chars=1200):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    p = cut.rfind(". ")
    return cut[: p + 1] if p > 200 else cut

def user_content(ex, retrieved=None):
    parts = []
    if retrieved:
        refs = "\n\n".join(f"Q: {r['question']}\nA: {truncate(r['rationale'], 500)}" for r in retrieved)
        parts.append(f"Related reference Q&A (may or may not be relevant):\n{refs}")
    if ex["qtype"] == "mcq":
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(ex["options"]))
        parts.append(
            "Answer the following multiple-choice medical question.\n\n"
            f"Question: {ex['question']}\n\nOptions:\n{opts}\n\n"
            "Reply with 'Answer: <letter>' followed by a short explanation."
        )
    else:
        ctx = ex.get("context") or ""
        if ctx and ctx.strip() != ex["question"].strip():
            parts.append(f"Context: {ctx}")
        parts.append(f"Question: {ex['question']}")
    return "\n\n".join(parts)

def build_messages(ex, retrieved=None):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(ex, retrieved)}
    ]

def build_target(ex):
    if ex["qtype"] == "mcq":
        target = f"Answer: {', '.join(ex['answer_letters'])}"
        if ex.get("rationale"):
            target += f"\nExplanation: {truncate(ex['rationale'])}"
        return target
    return truncate(ex["rationale"], 1500)
