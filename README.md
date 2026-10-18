# ClinicMate

ClinicMate answers English medical questions with a focus on healthcare in African settings. It is Qwen2.5-1.5B-Instruct fine-tuned with QLoRA on [AfriMed-QA](https://huggingface.co/datasets/intronhealth/afrimedqa_v2), with a DistilBERT classifier in front of it that refuses non-medical questions.

**Student research project. Not for clinical use and not medical advice.**

- Live demo: [add the Hugging Face Space link]
- Demo video: [add the video link]
- Report: [`report/ClinicMate_Report.pdf`](report/ClinicMate_Report.pdf)
- Models on the Hub: [`Tresor26/clinicmate-qwen2.5-1.5b`](https://huggingface.co/Tresor26/clinicmate-qwen2.5-1.5b) (generator) and [`Tresor26/clinicmate-domain-gate-v2`](https://huggingface.co/Tresor26/clinicmate-domain-gate-v2) (domain classifier)

## The problem

Most medical QA benchmarks come from US, Chinese and Indian exams, so they say little about how a model handles diseases, drugs and health-system constraints common in African countries. AfriMed-QA (Nimo et al., 2025) fills that gap with about 15,000 questions from African medical schools and crowd workers, and its authors found that small models struggle on it. This project asks how far a 1.5B model can get when fine-tuned on the dataset's own questions on a free GPU, and what it needs around it to be deployed.

## How it works

```
 user question
      │
      ▼
┌────────────────────────┐  P(out of domain) > 0.5  ┌──────────────────┐
│ Domain classifier       │ ───────────────────────► │ polite refusal    │
│ DistilBERT (v2)         │                          └──────────────────┘
└──────────┬─────────────┘
           │ medical
           ▼
┌────────────────────────────────────────────┐
│ Generator: Qwen2.5-1.5B-Instruct            │
│ + QLoRA adapter (r = 16), merged to fp16    │
└────────────────────────────────────────────┘
           │
           ▼
   streamed answer + disclaimer
```

## Dataset

[AfriMed-QA v2](https://huggingface.co/datasets/intronhealth/afrimedqa_v2) (CC-BY-SA-4.0, gated on Hugging Face). The file used is `afri_med_qa_15k_v2.4_phase_2_15275.csv`: 15,275 rows of MCQs (4,039), short-answer questions (1,236) and consumer queries (10,000). To reproduce, accept the dataset terms on Hugging Face, download the CSV and upload it as a private Kaggle dataset.

I did not use the official split. Its train split is entirely crowdsourced and its test split entirely expert-written, with only 129 training MCQs. I re-split the 5,234 MCQs and SAQs with gold answers 70/10/20, stratified by question type and tier (3,663 / 523 / 1,048). The evaluation set is the test SAQs plus the single-answer test MCQs: 739 MCQs and 243 SAQs. Consumer queries have no answers, so they are used only for the domain classifier and for qualitative checks. Results are therefore not comparable with the AfriMed-QA leaderboard. Experiment E4 uses the official crowdsourced-to-expert split.

## Results

Re-split evaluation set (739 MCQs, 243 SAQs). BERTScore is rescaled, so it can be negative.

| ID | System | MCQ accuracy [95% CI] | SAQ ROUGE-L | SAQ BERTScore |
|---|---|---|---|---|
| B0 | Random letter / majority letter (C) | 0.212 / 0.229 | | |
| B1 | TF-IDF nearest-neighbour retrieval | | 0.258 | 0.209 |
| B2 | Qwen2.5-0.5B zero-shot | 0.316 [0.283, 0.350] | 0.173 | 0.142 |
| B3 | Qwen2.5-1.5B zero-shot | 0.497 [0.460, 0.533] | 0.156 | 0.132 |
| E1 | 0.5B + QLoRA r=16 | 0.400 [0.365, 0.437] | 0.297 | 0.305 |
| E2 | 1.5B + QLoRA r=16 (deployed) | 0.541 [0.506, 0.578] | 0.292 | 0.302 |
| E3 | 1.5B + QLoRA r=8 | 0.548 [0.513, 0.586] | 0.302 | 0.316 |
| E5 | E2 + MiniLM retrieval (k=2) | 0.525 [0.488, 0.562] | 0.323 | 0.339 |
| | E2 merged into fp16 (what the app runs) | 0.549 [0.516, 0.587] | 0.312 | 0.334 |

McNemar tests on the MCQs: B3 vs E2 p = 0.015, E1 vs E2 p < 0.001, E2 vs E3 p = 0.59.

Official split (trained on 903 crowdsourced examples, tested on 3,930 expert questions):

| ID | System | MCQ accuracy [95% CI] | SAQ ROUGE-L | SAQ BERTScore |
|---|---|---|---|---|
| B3 | 1.5B zero-shot | 0.474 [0.458, 0.492] | 0.101 | -0.045 |
| E4 | 1.5B + QLoRA on official train | 0.484 [0.469, 0.501] | 0.129 | -0.012 |

Domain classifier (the challenge set is 60 questions I wrote by hand, 30 medical and 30 tricky non-medical):

| Classifier | Test accuracy (v2) | Challenge accuracy | Non-medical refused | Medical refused |
|---|---|---|---|---|
| TF-IDF + logistic regression, v2 data | 0.972 | 0.833 | 83.3% | 16.7% |
| DistilBERT, v2 data (deployed) | 0.990 | 0.783 | 56.7% | 0.0% |

Human review of 50 answers from E2 (30 SAQs, 20 consumer queries; one rater without medical training): mean correctness 2.92 out of 5, at least one hallucinated claim in 50%, potentially harmful in 10%. The full analysis is in the report.

## Repository layout

```
src/            shared code: cleaning and parsing (data.py), prompts (prompts.py), scoring (evaluation.py)
notebooks/      01 to 08, run in order on Kaggle
results/        metrics (eval_*.json), predictions (preds_*.jsonl), figures, rating sheets
app/            Gradio app for Hugging Face Spaces (app.py, prompts.py, requirements.txt)
report/         project report
gate_challenge.csv   hand-written challenge set for the domain classifier
requirements.txt     package versions used on Kaggle
```

## Reproducing

All notebooks ran on Kaggle with one Tesla T4. The first cell of each notebook clones this repository, adds it to the Python path and logs in to Hugging Face with a Kaggle secret called `HF_TOKEN`. Trained models are pushed to private Hub repos, and later notebooks load them from there. Each notebook reads the outputs of earlier ones through Kaggle's "Add Input".

| Notebook | What it does |
|---|---|
| `01-eda` | Explores the raw CSV |
| `02-data-preparation` | Cleans, filters, deduplicates and splits the data into `data/*.jsonl` |
| `03-baselines` | Random, majority and TF-IDF retrieval baselines |
| `04-domain-gate-` | Builds the classifier data and trains TF-IDF and DistilBERT classifiers (v1 and v2) |
| `05-E3` | QLoRA training. Set `RUN` to `E2_qwen1.5b_r16`, `E1_qwen0.5b_r16`, `E4_qwen1.5b_official` or `E3_qwen1.5b_r8` |
| `06-evaluation` | Evaluates every system, McNemar tests, out-of-domain test, human-rating sheet, retrieval (E5) |
| `07-error-analysis` | Accuracy by specialty, tier, country and length, letter bias, SAQ error categories |
| `08-merge-and-export` | Merges the E2 adapter into fp16, re-evaluates it, compares decoding settings, writes the model card |

Package versions are in [`requirements.txt`](requirements.txt). To run outside Kaggle, install them with `pip install -r requirements.txt` and change the paths and the `kaggle_secrets` login in the first cell of each notebook.

## Deployment

`app/` is the Gradio app on Hugging Face Spaces. It loads the merged model and the classifier from the Hub, refuses a question when the classifier's out-of-domain probability is above 0.5, and otherwise streams an answer with greedy decoding (repetition penalty 1.1, at most 256 new tokens) followed by a disclaimer.

Free Gradio Spaces run on ZeroGPU. On this Space the ZeroGPU worker could not initialise CUDA, so the Space sets the variable `FORCE_CPU=1` and the model runs on the Space's CPU in fp32. Without that variable, the app uses ZeroGPU or any dedicated GPU it finds.

## What I implemented and what I reused

I implemented:

- the data cleaning, option and answer parsing, filtering, deduplication and the stratified re-split (`src/data.py`, notebook 02)
- the prompt and training-target format, including label masking so the loss covers only the answer (`src/prompts.py`, notebook 05)
- MCQ scoring from the letter logits after "Answer:", the generation loop, bootstrap confidence intervals and McNemar comparisons (`src/evaluation.py`, notebook 06)
- the domain classifier datasets (filtered SQuAD, WebQuestions and Yahoo Answers negatives) and the 60-question challenge set
- the experiment design (E1 to E5), the error analysis, the hand labelling of errors and the 50-answer human review
- the Gradio app, including the domain check, streaming and the CPU fallback

I reused:

- the AfriMed-QA dataset, and SQuAD, WebQuestions and Yahoo Answers Topics from the Hugging Face Hub
- pre-trained models: Qwen2.5-0.5B/1.5B-Instruct, distilbert-base-uncased, all-MiniLM-L6-v2, and roberta-large inside BERTScore
- libraries: Hugging Face Transformers, PEFT, bitsandbytes, Datasets, Evaluate, sentence-transformers, scikit-learn, Gradio
- the QLoRA recipe (NF4 base, LoRA on all linear layers) from Dettmers et al. (2023)

## Limitations

- English only, and about half of the source rows come from Nigeria.
- Only 7.6% of MCQs have explanations, and consumer queries have no answers, so the model is weakest on lay-language questions.
- A 1.5B model still gets about 45% of held-out MCQs wrong and hallucinates in half of the reviewed answers.
- ROUGE-L and BERTScore reward matching the reference's length and wording, not clinical correctness.
- The human review had one rater without medical training.
- The domain classifier still lets through non-medical questions that use medical words, such as "how do I remove a virus from my laptop".

## Citations

- Nimo, C., Olatunji, T., et al. (2025). AfriMed-QA: A pan-African, multi-specialty, medical question-answering benchmark dataset. *ACL 2025*. https://doi.org/10.18653/v1/2025.acl-long.96
- Qwen Team. (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
- Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.
- Lin, C.-Y. (2004). ROUGE. *Text Summarization Branches Out*.
- Zhang, T., et al. (2020). BERTScore. *ICLR*.

The full reference list is in the report.

## Licence

The fine-tuned model is released under CC-BY-SA-4.0, inherited from AfriMed-QA. Qwen2.5 is Apache-2.0.
