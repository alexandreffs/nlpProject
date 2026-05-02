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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# DECODER_MODEL = "google/medgemma-4b-it"

# tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)

# decoder_model = AutoModelForCausalLM.from_pretrained(
#     DECODER_MODEL,
#     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )

# def generate_answer(question, reference_docs, max_new_tokens=120):
#     reference_context = build_reference_context(reference_docs)
#     prompt = build_answer_prompt(question, reference_context)

#     messages = [
#         {
#             "role": "system",
#             "content": "You are a biomedical RAG assistant. Answer only using the provided references."
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]

#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt"
#     )

#     inputs = {
#         k: v.to(decoder_model.device)
#         for k, v in inputs.items()
#     }

#     input_len = inputs["input_ids"].shape[-1]

#     with torch.inference_mode():
#         generation = decoder_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             repetition_penalty=1.15,
#             no_repeat_ngram_size=4,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     generated_tokens = generation[0][input_len:]
#     response = tokenizer.decode(
#         generated_tokens,
#         skip_special_tokens=True
#     ).strip()

#     return response

DECODER_MODEL = "google/medgemma-4b-it"
pipe = pipeline(
    "text-generation",
    model=DECODER_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    tokenizer=DECODER_MODEL
)

def generate_answer(question, reference_docs, max_new_tokens=120):
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

    # Some chat pipelines return list of messages, others return plain text
    if isinstance(response, list):
        response = response[-1]["content"]

    response = response.strip()

    return response