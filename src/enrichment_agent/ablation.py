"""
HERMES — Ablation Baseline
============================
Evalúa técnicas de clasificación sin pipeline CPM para comparar con Hermes.

CONFIGURACIÓN RÁPIDA:
  - TECHNIQUE: "zero_shot" o "cot"
  - DATASET: "semeval2017" o "semeval2018"

El sistema de checkpoint permite retomar si el proceso se interrumpe.
"""

import asyncio
import json
import os
import re
import time

import openpyxl
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


# ============================================================
# CONFIGURACIÓN PRINCIPAL — CAMBIAR AQUÍ
# ============================================================

TECHNIQUE = "zero_shot"   # "zero_shot" o "cot"
DATASET   = "semeval2018" # "semeval2017" o "semeval2018"

MAX_COMMENTS = None  # None = todos los comentarios

MAX_RETRIES = 3
BASE_DELAY = 2  # segundos, se duplica en cada retry


# ============================================================
# CONFIGURACIÓN POR DATASET
# ============================================================

DATASET_CONFIG = {
    "semeval2017": {
        "format": "xlsx",
        "path": "enrichment_agent/Datasets/SemEval_Dataset_Unido.xlsx",
        "valid_labels": {"positive", "negative", "neutral"},
        "label_aliases": {
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral",
            "neut": "neutral",
        },
        "task_description": (
            "Classify the sentiment expressed in this tweet as exactly one of: "
            "positive, negative, or neutral."
        ),
        "output_format": (
            '{"sentiment_label": "positive"}\n'
            'or\n'
            '{"sentiment_label": "negative"}\n'
            'or\n'
            '{"sentiment_label": "neutral"}'
        ),
    },
    "semeval2018": {
        "format": "tsv",
        "input_path": "enrichment_agent/Datasets/SemEval2018-T3_input_test_taskA_emoji.txt",
        "gold_path":  "enrichment_agent/Datasets/SemEval2018-T3_gold_test_taskA_emoji.txt",
        "label_map": {"0": "not_irony", "1": "irony"},
        "valid_labels": {"irony", "not_irony"},
        "label_aliases": {
            "not irony":  "not_irony",
            "non-irony":  "not_irony",
            "no irony":   "not_irony",
            "ironic":     "irony",
            "non_ironic": "not_irony",
        },
        "task_description": (
            "Classify whether this tweet is ironic or not ironic."
        ),
        "output_format": (
            '{"sentiment_label": "irony"}\n'
            'or\n'
            '{"sentiment_label": "not_irony"}'
        ),
    },
}


# ============================================================
# PROMPTS POR TÉCNICA
# ============================================================

def build_prompt(technique: str, dataset: str, tweet: str) -> str:
    cfg  = DATASET_CONFIG[dataset]
    task = cfg["task_description"]
    fmt  = cfg["output_format"]

    if technique == "zero_shot":
        return f"""{task}

Tweet: {tweet}

Respond with ONLY a JSON object in this exact format, nothing else:
{fmt}
"""

    elif technique == "cot":
        return f"""{task}

Tweet: {tweet}

After your reasoning, output ONLY a valid JSON object in this exact format:
{fmt}

Let's think step by step.
"""

    else:
        raise ValueError(f"Unknown technique: '{technique}'. Use 'zero_shot' or 'cot'.")


# ============================================================
# NOMBRE DEL ARCHIVO DE OUTPUT
# ============================================================

def get_output_filename(technique: str, dataset: str) -> str:
    return f"ablation_{technique}_{dataset}.json"


# ============================================================
# DATASET LOADERS
# ============================================================

def load_comments_xlsx(dataset: str, limit: int | None = None) -> list[dict]:
    """Loader para SemEval 2017 — archivo Excel."""
    cfg          = DATASET_CONFIG[dataset]
    valid_labels = cfg["valid_labels"]

    wb = openpyxl.load_workbook(cfg["path"], read_only=True)
    ws = wb.active

    comments = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
        tweet_id, gold_label, tweet_text = row[0], row[1], row[2]
        gold_label = str(gold_label).strip().lower()

        if gold_label not in valid_labels:
            print(f"Skipping row {i}: unknown label '{gold_label}'")
            continue

        comments.append({
            "index":      i,
            "tweet_id":   str(tweet_id),
            "gold_label": gold_label,
            "text":       str(tweet_text).strip(),
        })

        if limit and len(comments) >= limit:
            break

    wb.close()
    return comments


def load_comments_tsv(dataset: str, limit: int | None = None) -> list[dict]:
    """Loader para SemEval 2018 — dos archivos TSV (input + gold)."""
    cfg          = DATASET_CONFIG[dataset]
    valid_labels = cfg["valid_labels"]
    label_map    = cfg["label_map"]

    # Leer gold labels: tweet_id -> label
    gold = {}
    with open(cfg["gold_path"], "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tweet_id, label_num = parts[0], parts[1]
            label = label_map.get(label_num.strip())
            if label is None:
                print(f"Skipping gold row {tweet_id}: unknown label '{label_num}'")
                continue
            gold[tweet_id] = label

    # Leer tweets y combinar con gold
    comments = []
    with open(cfg["input_path"], "r", encoding="utf-8") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            tweet_id, tweet_text = parts[0], parts[1]

            gold_label = gold.get(tweet_id)
            if gold_label not in valid_labels:
                print(f"Skipping tweet {tweet_id}: missing or invalid gold label")
                continue

            comments.append({
                "index":      i,
                "tweet_id":   tweet_id,
                "gold_label": gold_label,
                "text":       tweet_text.strip(),
            })

            if limit and len(comments) >= limit:
                break

    return comments


def load_comments(dataset: str, limit: int | None = None) -> list[dict]:
    """Dispatcher que selecciona el loader correcto según el dataset."""
    fmt = DATASET_CONFIG[dataset]["format"]
    if fmt == "xlsx":
        return load_comments_xlsx(dataset, limit)
    elif fmt == "tsv":
        return load_comments_tsv(dataset, limit)
    else:
        raise ValueError(f"Unknown format: '{fmt}'")


# ============================================================
# MODELO
# ============================================================

def init_model() -> ChatGroq:
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ============================================================
# PARSEO DE RESPUESTA
# ============================================================

def parse_response(text: str, dataset: str) -> str:
    cfg           = DATASET_CONFIG[dataset]
    valid_labels  = cfg["valid_labels"]
    label_aliases = cfg["label_aliases"]

    text = text.strip()

    # Intentar parseo JSON — tomar el último bloque JSON (útil para CoT)
    try:
        clean        = text.replace("```json", "").replace("```", "").strip()
        json_matches = re.findall(r'\{[^{}]+\}', clean)
        if json_matches:
            data  = json.loads(json_matches[-1])
            label = data.get("sentiment_label", "").strip().lower()
            if label in valid_labels:
                return label
            mapped = label_aliases.get(label)
            if mapped and mapped in valid_labels:
                return mapped
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: buscar etiqueta válida en el texto
    text_lower = text.lower()
    for label in valid_labels:
        if label in text_lower:
            return label

    # Fallback: aliases
    for alias, mapped in label_aliases.items():
        if alias in text_lower and mapped in valid_labels:
            return mapped

    return ""


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(output_file: str) -> tuple[list[dict], set[int]]:
    if not os.path.exists(output_file):
        return [], set()

    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    processed = {r["index"] for r in results}
    print(f"Checkpoint cargado: {len(results)} tweets ya procesados")
    return results, processed


# ============================================================
# RETRY
# ============================================================

def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    signals   = ["429", "rate limit", "500", "502", "503", "504",
                 "server error", "timeout", "timed out", "overloaded"]
    return any(s in error_str for s in signals)


async def invoke_with_retry(model: ChatGroq, prompt: str) -> str:
    msg = [HumanMessage(content=prompt)]
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await model.ainvoke(msg)
            return response.content if isinstance(response.content, str) else ""
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  Retryable error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

async def run_ablation():
    output_file = get_output_filename(TECHNIQUE, DATASET)

    print(f"\n{'='*60}")
    print(f"  HERMES Ablation Baseline")
    print(f"  Técnica  : {TECHNIQUE.upper()}")
    print(f"  Dataset  : {DATASET}")
    print(f"  Output   : {output_file}")
    print(f"{'='*60}\n")

    # Cargar dataset
    comments = load_comments(DATASET, limit=MAX_COMMENTS)
    cfg       = DATASET_CONFIG[DATASET]
    dist      = {l: sum(1 for c in comments if c["gold_label"] == l) for l in cfg["valid_labels"]}
    print(f"Cargados {len(comments)} tweets")
    print(f"  Distribución: {dist}")

    # Cargar checkpoint
    results, processed_indices = load_checkpoint(output_file)
    remaining = [c for c in comments if c["index"] not in processed_indices]

    if not remaining:
        print("\nTodos los tweets ya fueron procesados.")
        return

    if processed_indices:
        print(f"Retomando: {len(remaining)} tweets pendientes")

    model       = init_model()
    batch_start = time.time()

    for i, comment in enumerate(remaining):
        progress = len(processed_indices) + i + 1
        total    = len(comments)
        print(f"\n[{progress}/{total}] {comment['text'][:80]}...")

        tweet_start = time.time()

        try:
            prompt       = build_prompt(TECHNIQUE, DATASET, comment["text"])
            raw_response = await invoke_with_retry(model, prompt)
            elapsed      = time.time() - tweet_start
            prediction   = parse_response(raw_response, DATASET)

            if prediction:
                print(f"  -> {prediction}  ({elapsed:.1f}s)")
            else:
                print(f"  -> No se pudo parsear: {raw_response[:100]}  ({elapsed:.1f}s)")

            result = {
                "index":      comment["index"],
                "tweet_id":   comment["tweet_id"],
                "comment":    comment["text"],
                "gold_label": comment["gold_label"],
                "raw_response": raw_response,
                "convergence": {
                    "sentiment_label": prediction,
                    "reasoning": f"Ablation {TECHNIQUE} — sin pipeline CPM.",
                },
                "technique":       TECHNIQUE,
                "dataset":         DATASET,
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = time.time() - tweet_start
            print(f"  Error ({elapsed:.1f}s): {e}")
            result = {
                "index":      comment["index"],
                "tweet_id":   comment["tweet_id"],
                "comment":    comment["text"],
                "gold_label": comment["gold_label"],
                "error":      str(e),
                "technique":  TECHNIQUE,
                "dataset":    DATASET,
                "elapsed_seconds": round(elapsed, 2),
            }

        results.append(result)

        # Guardar progresivamente
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    batch_elapsed = time.time() - batch_start
    print(f"\nAblación completa. Resultados en: {output_file}")
    print(f"  Procesados esta corrida : {len(remaining)} tweets en {batch_elapsed:.1f}s")
    print(f"  Total en archivo        : {len(results)} tweets")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_ablation())