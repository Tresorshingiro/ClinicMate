import os

ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))      # set only on ZeroGPU hardware
FORCE_CPU = bool(os.getenv("FORCE_CPU"))          # Space variable: never request a GPU
if ZERO_GPU:
    import spaces       # must be imported before torch/transformers so it can patch CUDA first

from threading import Thread

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline

from prompts import SYSTEM_PROMPT, REFUSAL, DISCLAIMER

gpu = spaces.GPU(duration=60) if ZERO_GPU and not FORCE_CPU else (lambda f: f)
if not FORCE_CPU and (ZERO_GPU or torch.cuda.is_available()):     # ZeroGPU or a dedicated GPU Space
    DEVICE, DTYPE = "cuda", torch.float16
else:
    DEVICE, DTYPE = "cpu", torch.float32   # CPUs are slow or unsupported with fp16

if ZERO_GPU and FORCE_CPU:
    @spaces.GPU
    def _unused():          # ZeroGPU Spaces won't start without at least one @spaces.GPU function
        pass

MODEL_ID = "Tresor26/clinicmate-qwen2.5-1.5b"     # E2 merged into fp16 (notebook 08)
GATE_ID = "Tresor26/clinicmate-domain-gate-v2"    # DistilBERT gate v2 (notebook 04)
GATE_THRESHOLD = 0.5      # refuse if P(out_of_domain) is above this
GEN_KWARGS = dict(max_new_tokens=256, do_sample=False, repetition_penalty=1.1)  # chosen in notebook 08

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
model.eval()
gate = pipeline("text-classification", model=GATE_ID, top_k=None, device=-1)   # small: keep it on CPU


def p_out_of_domain(text):
    scores = {d["label"]: d["score"] for d in gate([text], truncation=True)[0]}
    return scores["out_of_domain"]


@gpu
def generate(question):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"}]      # same format as SAQ training
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, **GEN_KWARGS)).start()
    out = ""
    for chunk in streamer:
        out += chunk
        yield out
    yield out + DISCLAIMER


def answer(question):
    question = (question or "").strip()
    if not question:
        yield "Please type a question."
        return
    p_ood = p_out_of_domain(question)
    if p_ood > GATE_THRESHOLD:              # the generator is never called for refused questions
        yield f"{REFUSAL}\n\n*(domain gate: P(out-of-domain) = {p_ood:.2f})*"
        return
    yield from generate(question)


EXAMPLES = [
    "What is the first-line treatment for uncomplicated malaria in adults?",
    "My child has had diarrhoea for 3 days. What danger signs should I look for?",
    "How is sickle cell disease inherited?",
    "Who won the 2010 FIFA World Cup?",          # shows the refusal
]

with gr.Blocks(title="ClinicMate") as demo:
    gr.Markdown("# ClinicMate\nA medical question-answering assistant for African healthcare "
                "contexts, fine-tuned on AfriMed-QA. **Student research project, not medical advice.**")
    q = gr.Textbox(label="Your health question", lines=3)
    btn = gr.Button("Ask", variant="primary")
    out = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=q)
    btn.click(answer, inputs=q, outputs=out)
    q.submit(answer, inputs=q, outputs=out)

demo.queue().launch()
