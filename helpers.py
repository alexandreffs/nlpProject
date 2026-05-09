import json
from transformers import pipeline
import torch


def load_corpus_txt(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            pmid = str(obj["id"])
            contents = obj["contents"].strip()

            parts = contents.split("\n", 1)
            title = parts[0].strip()
            abstract = parts[1].strip() if len(parts) > 1 else ""

            docs.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract
            })
    return docs

######################################################################################

# ============================================================
# Build context for answer generation
# ============================================================

def build_reference_context(reference_docs):
    lines = []

    for doc in reference_docs:
        pmid = doc["pmid"]
        for sent_obj in doc["selected_sentences"]:
            sent = sent_obj["sentence"]
            lines.append(f"[PMID:{pmid}] {sent}")

    return "\n".join(lines)

# ============================================================
# Prompt for cited biomedical answer generation
# ============================================================

def build_answer_prompt(question, reference_context):
    return f"""
You are a biomedical RAG assistant.

Use ONLY the reference sentences below.
Do NOT add any fact that is not explicitly stated in the references.
If the references do not answer part of the question, say that the evidence is insufficient.

Write a short patient-friendly answer.
Maximum 250 words.
Every sentence must contain at least one citation in the format [PMID:123456].
Do not use more than 3 PMIDs per sentence.

Question:
{question}

References:
{reference_context}

Answer:
""".strip()

# ============================================================
# Generate answer helper
# ============================================================

DECODER_MODEL = "google/medgemma-4b-it"
pipe = pipeline(
    "text-generation",
    model=DECODER_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    tokenizer=DECODER_MODEL
)

def generate_answer(question, reference_docs, max_new_tokens=400):
    reference_context = build_reference_context(reference_docs)
    prompt = build_answer_prompt(question, reference_context)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False
    )

    response = output[0]["generated_text"]

    if isinstance(response, list):
        response = response[-1]["content"]

    response = response.strip()

    return response