# Absolute Grading: Outputs score of 1 to 5

import os
import json
import pandas as pd
from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from transformers import AutoTokenizer

# Set HuggingFace token
os.environ["OPENAI_API_BASE"] = "https://my-vLLM-api-endpoint.net/v1" # Replace with your actual vLLM API endpoint

# Load the grading rubric
with open("score_rubrics_en.json", "r", encoding="utf-8") as f:
    score_rubrics = json.load(f)

# Prepare all formatted rubrics once before the main loop
formatted_rubrics = [
    SCORE_RUBRIC_TEMPLATE.format(**rubric)
    for rubric in score_rubrics
]

# Read prompt from file
with open("public_dataset_without_instruction-prompt.txt", "r", encoding="utf-8") as f:
    prompt_without_instruction = f.read()

with open("public_dataset_full-prompt.txt", "r", encoding="utf-8") as f:
    prompt_with_instruction = f.read()

models = [
    {
        "model": "smollm2:135m",
        "intervention": False,
    },
    {
        "model": "llama3.2:1b",
        "intervention": True,
    },
    {
        "model": "llama3.2:3b",
        "intervention": False,
    },
    {
        "model": "qwen3:4b",
        "intervention": True,
    },
    {
        "model": "qwen3:14b",
        "intervention": False,
    },
    {
        "model": "deepseek-r1:14b",
        "intervention": True,
    },
    {
        "model": "mistral-small:22b",
        "intervention": False,
    },
    {
        "model": "deepseek-r1:32b",
        "intervention": True,
    },
    {
        "model": "llama3.3:70b",
        "intervention": False,
    },
    {
        "model": "deepseek-v3.1:671b-cloud",
        "intervention": True,
    }
]

# Load responses
responses_path = os.path.join(os.getcwd(), "responses", "all_responses_joined.csv")
df = pd.read_csv(responses_path)

# Load reference answer
with open("reference_answer.md", "r", encoding="utf-8") as f:
    reference_answer = f.read()

# Prepare model and judge
model = LiteLLM('hosted_vllm/Unbabel/M-Prometheus-14B') # Adapt model as desired
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

# Prepare output
grading_rows = []

for rubric_idx, rubric in enumerate(score_rubrics, start=1):
    instructions = []
    responses = []
    reference_answers = []
    meta_rows = []

    for idx, row in df.iterrows():
        phase = row.get("phase", "")
        model_name = row.get("model-name", "")
        model_config = next((m for m in models if m["model"] == model_name), None)
        intervention = model_config["intervention"] if model_config else False
        intervention_for_row = True if phase == 2 and intervention else False

        if intervention_for_row:
            instruction = prompt_with_instruction
        else:
            instruction = prompt_without_instruction

        response = row.get("response", "")
        
        instructions.append(instruction)
        responses.append(response)
        reference_answers.append(reference_answer)
        meta_rows.append({
            "model-name": row.get("model-name", ""),
            "phase": row.get("phase", ""),
            "attempt": row.get("attempt", ""),
            "intervention": intervention_for_row
        })

    # Token counting and filtering
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

    def count_tokens_hf(*args):
        text = " ".join(str(x) for x in args)
        return len(tokenizer.encode(text))

    MAX_TOKENS = 32768 - 2048 # Reserve tokens for model output

    # Filter out inputs that exceed the token limit
    filtered_instructions = []
    filtered_responses = []
    filtered_reference_answers = []
    filtered_meta_rows = []

    for inst, resp, ref, meta in zip(instructions, responses, reference_answers, meta_rows):
        total_tokens = count_tokens_hf(inst, resp, SCORE_RUBRIC_TEMPLATE.format(**rubric), ref)
        if total_tokens <= MAX_TOKENS:
            filtered_instructions.append(inst)
            filtered_responses.append(resp)
            filtered_reference_answers.append(ref)
            filtered_meta_rows.append(meta)
        else:
            # Optionally, log or store an error for this row
            meta_copy = meta.copy()
            meta_copy[f"feedback_criteria{rubric_idx}"] = "Skipped: input too long"
            meta_copy[f"score_criteria{rubric_idx}"] = "Error"
            grading_rows.append(meta_copy)

    # Now batch grade only the filtered inputs
    try:
        feedbacks, scores = judge.absolute_grade(
            instructions=filtered_instructions,
            responses=filtered_responses,
            rubric=SCORE_RUBRIC_TEMPLATE.format(**rubric),
            reference_answers=filtered_reference_answers,
            params={
                "max_tokens": 2048,
                "repetition_penalty": 1.03,
                "best_of": 1,
                "temperature": 1.0,
                "top_p": 0.9,
            }
        )
    except Exception as e:
        feedbacks = [f"Error: {str(e)}"] * len(filtered_responses)
        scores = ["Error"] * len(filtered_responses)

    # Store results for this rubric
    for meta, feedback, score in zip(filtered_meta_rows, feedbacks, scores):
        grading_row = next((r for r in grading_rows if
                            r["model-name"] == meta["model-name"] and
                            r["phase"] == meta["phase"] and
                            r["attempt"] == meta["attempt"] and
                            r["intervention"] == meta["intervention"]), None)
        if grading_row is None:
            grading_row = meta.copy()
            grading_rows.append(grading_row)
        grading_row[f"feedback_criteria{rubric_idx}"] = feedback
        grading_row[f"score_criteria{rubric_idx}"] = score

grading_df = pd.DataFrame(grading_rows)
grading_df.to_csv(
    os.path.join(os.getcwd(), "evaluation", "all_responses_grading_separated_vllm_with_reference.csv"),
    index=False
)