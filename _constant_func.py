"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
from pydantic import BaseModel
import json
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import os


# --------------------------------------------------------------------------- #
#  PROMPT TEMPLATES                                                            #
# --------------------------------------------------------------------------- #

prompt_template = """
You are an expert systematic literature reviewer specializing in Open Source Software (OSS) sustainability research.

Your task is to extract ACTIONABLE RECOMMENDATIONS from the article text provided below.

STRICT DEFINITION — An actionable recommendation MUST satisfy ALL of the following:
  1. It is a concrete, specific action that a person or team can directly implement.
  2. It is expressed as an imperative or directive (e.g., "use X", "adopt Y", "implement Z").
  3. It is distinct from a general observation, finding, or description of current practice.
  4. It pertains to improving the sustainability, contribution, or maintenance of OSS projects.

DO NOT extract:
  - General observations or descriptions (e.g., "projects tend to have more bugs")
  - Vague suggestions without a clear action (e.g., "more work is needed")
  - Future research directions (e.g., "future studies should explore X")
  - Statements about what the authors did (e.g., "we analyzed 100 projects")

EXTRACTION RULES:
  - Extract ONLY recommendations that are explicitly stated or very directly implied in the text.
  - Do NOT infer, speculate, or generalize beyond what is written.
  - Each recommendation must be a single, self-contained action — split compound recommendations.
  - Keep each field concise: 1–3 sentences maximum per field.
  - For 'positive_impact': quote or closely paraphrase only what the article explicitly states will improve.
  - For 'evidence': cite the specific data, experiment, or finding in the article that justifies the recommendation. Use exact figures or results where available.
  - For 'confidence': score 0.9–1.0 only if the article explicitly states the recommendation with clear evidence; 0.7–0.89 if implied with some evidence; below 0.7 if weakly supported.

OUTPUT: Return ONLY a valid JSON object. No preamble, no explanation, no markdown fences.

If no actionable recommendations exist in the text, return exactly:
{"message": "NO ACTIONABLE CAN BE DERIVED"}

Full text of the article:
"""


IP_template = """
Using the definition and rules provided above, extract all actionable recommendations from the article.

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""


CoT_template = """
Using the definition and rules provided above, extract actionable recommendations by working through these steps internally before producing output:

STEP 1 — SCAN: Read every sentence. Flag only sentences containing imperative directives or explicit suggestions targeted at practitioners.
STEP 2 — FILTER: Discard any flagged sentence that is a general observation, future work direction, or lacks a specific implementable action.
STEP 3 — DEDUPLICATE: Merge recommendations that describe the same action even if worded differently.
STEP 4 — ENRICH: For each surviving recommendation, locate the specific impact claim and supporting evidence in the article text. Use exact numbers or findings where present.
STEP 5 — SCORE: Assign confidence based strictly on explicitness (0.9–1.0 = explicitly stated with evidence, 0.7–0.89 = implied with partial evidence, <0.7 = weakly supported).

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""


RA_template = """
Using the definition and rules provided above, extract actionable recommendations using this retrieval-augmented reasoning approach:

For EACH candidate recommendation:

  [CHECK 1 — ACTIONABILITY]
  Is this a concrete, implementable action (not an observation or future work)?
  → If NO: discard. If YES: continue.

  [CHECK 2 — DIRECTNESS]
  Is this recommendation explicitly stated in the article (not inferred by you)?
  → If NO: discard. If YES: continue.

  [CHECK 3 — IMPACT]
  Does the article explicitly state what will improve if this is adopted?
  → If YES: extract the exact claim. If NO: set "NO IMPACT FOUND".

  [CHECK 4 — EVIDENCE]
  Does the article provide a specific data point, study result, or empirical finding supporting this recommendation?
  → If YES: extract it verbatim or near-verbatim. If NO: set "NO EVIDENCE FOUND".

  [CHECK 5 — CONFIDENCE]
  Rate confidence 0.9–1.0 (explicit + evidence), 0.7–0.89 (implied + partial evidence), <0.7 (weak).

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""


prompt_template_for_assigning_categories = """
You are a systematic literature reviewer analyzing practical suggestions—called actionables—to enhance the sustainability of Open Source Software (OSS) projects.

You are given:
  i)  A single actionable recommendation.
  ii) A fixed set of categories, each with a definition and criteria.

Your task: Assign the SINGLE most appropriate criterion label to the actionable.

RULES:
  - Output ONLY the criterion label (e.g., "a", "b", "c") under the best-matching category.
  - Do NOT output reasoning, explanation, category names, or any other text.
  - If multiple criteria seem plausible, choose the most specific one.

CATEGORIES AND CRITERIA:

1. New Contributor Onboarding and Involvement
   Definition: Ensuring new contributors can easily join, understand, and meaningfully contribute.
   a) Facilitates integration of new contributors via mentorship, onboarding materials, or simplified contribution process.
   b) Improves documentation or support mechanisms for first-time contributors.
   c) Builds a welcoming, inclusive culture for new participants.

2. Code Standards and Maintainability
   Definition: Ensuring the codebase adheres to standards for readability, modularity, and maintainability.
   a) Improves quality, readability, or structure of the codebase.
   b) Enforces coding guidelines, refactors for maintainability, or reduces technical debt.
   c) Uses linters, formatters, or static code analysis tools.

3. Automated Testing and Quality Assurance
   Definition: Ensuring robustness through automated testing and quality assurance.
   a) Implements or improves automated testing frameworks and strategies.
   b) Detects bugs early and ensures high-quality releases.

4. Community Collaboration and Engagement
   Definition: Fostering collaboration, communication, and engagement within the OSS community.
   a) Improves communication between contributors, maintainers, and users.
   b) Organizes community events, discussions, or collaboration platforms.
   c) Improves community governance and decision-making tools and processes.

5. Documentation Practices
   Definition: Ensuring documentation is thorough, up-to-date, and accessible.
   a) Improves quality, clarity, or accessibility of documentation (user guides, API refs, contributor guides).
   b) Keeps documentation synchronized with the codebase.
   c) Makes documentation accessible to non-expert or non-English audiences.

6. Project Management and Governance
   Definition: Governance structures and project management for long-term sustainability.
   a) Enhances governance model, clarifies roles, or improves decision-making.
   b) Defines processes for issue triaging, release management, or conflict resolution.
   c) Improves transparency of project goals, progress, and decisions.

7. Security Best Practices and Legal Compliance
   Definition: Securing the project and ensuring legal/regulatory compliance.
   a) Improves security posture via best practices, vulnerability fixes, or audits.
   b) Ensures license compliance, CLAs, or data privacy alignment.
   c) Implements dependency management, security audits, or secure coding practices.

8. CI/CD and DevOps Automation
   Definition: Automating build, test, and deployment pipelines.
   a) Sets up or enhances CI/CD pipelines for faster, reliable releases.
   b) Automates infrastructure provisioning, containerization, or cloud deployment.
   c) Integrates DevOps practices for repeatable software development and deployment.

ACTIONABLE:
"""


# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                   #
# --------------------------------------------------------------------------- #

allowable_models = [
    "meta.llama3-1-405b-instruct-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-v2",
    "meta.llama3-1-70b-instruct-v1:0",
    "deepseek-r1:1.5b",
    "llama3.2:3b-instruct-q4_K_M",
    "mixtral:8x7b-instruct-v0.1-q4_K_M",
    "qordmlwls/llama3.1-medical:latest",
    "medllama2:latest",
    "meditron:70b",
    "llama3.2:latest",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "tinyllama",
    "llama3.3:70b",
    "qwen2.5:72b",
    "deepseek-r1:7b",
    "thewindmom/llama3-med42-8b:latest",
    "mixtral:latest",
    "hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m"
]

allowable_prompting_methods = ["IP", "CoT", "RA"]


# --------------------------------------------------------------------------- #
#  UTILITY FUNCTIONS                                                           #
# --------------------------------------------------------------------------- #

def remove_newlines(text):
    return text.replace("\n", "").replace("\r", "")


def clean_response(response):
    """Extract and parse the first valid JSON object from a raw LLM response."""
    json_start_index = response.find("{")
    json_end_index   = response.rfind("}")

    if json_start_index == -1 or json_end_index == -1:
        print("No JSON braces found in response.")
        return {"message": "NO ACTIONABLE CAN BE DERIVED"}

    json_text = response[json_start_index:json_end_index + 1]

    # Normalize angle-bracket placeholders: <value> → "value"
    json_text = re.sub(r':\s*<([^>]+)>', r': "\1"', json_text)

    # Remove markdown fences if model ignored the instruction
    json_text = re.sub(r'```(?:json)?', '', json_text).strip()

    json_part = re.search(r'\{.*\}', json_text, re.DOTALL)

    if json_part:
        try:
            parsed_json = json.loads(json_part.group())
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
    else:
        print("No valid JSON object found in response.")

    return {"message": "NO ACTIONABLE CAN BE DERIVED"}


def get_pdfs(local_dir):
    pdf_files = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def extract_article_title(markdown_text):
    """Extracts all level-1 headings from a Markdown string."""
    return re.findall(r'^# (.+)', markdown_text, re.MULTILINE)


def clean_article_text(text):
    """Removes everything from ACKNOWLEDGMENTS or REFERENCES onward."""
    pattern = r'(?i)### (ACKNOWLEDGMENTS?|REFERENCES?).*'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def extract_article_link(text):
    """Extracts DOI or article URLs from text."""
    url_pattern = r'https?://[^\s\)]+'
    return re.findall(url_pattern, text)


# --------------------------------------------------------------------------- #
#  EVALUATION UTILITIES                                                        #
# --------------------------------------------------------------------------- #

class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int


def extract_first_binary(var_name):
    match = re.search(r'[01]', var_name)
    return match.group(0) if match else 0


def fix_json(json_input, response):
    if isinstance(json_input, dict):
        return json_input

    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"reason_for_the_label": response, "label": extract_first_binary(response)}

    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json
        return {"reason_for_the_label": response, "label": extract_first_binary(response)}
    except json.JSONDecodeError:
        pass

    for i in range(len(json_input), 0, -1):
        try:
            parsed_json = json.loads(json_input[:i])
            if isinstance(parsed_json, dict):
                return parsed_json
        except json.JSONDecodeError:
            continue

    return {"reason_for_the_label": response, "label": extract_first_binary(response)}


def get_dataframes(metrics, modality):
    return pd.DataFrame({
        'Modality':              [modality],
        'TP':                    [metrics['TP']],
        'TN':                    [metrics['TN']],
        'FP':                    [metrics['FP']],
        'FN':                    [metrics['FN']],
        'Sensitivity':           [metrics['Sensitivity']],
        'Specificity':           [metrics['Specificity']],
        'Precision':             [metrics['Precision']],
        'F1-Score':              [metrics['F1-Score']],
        'FPR':                   [metrics['FPR']],
        'TPR':                   [metrics['TPR']],
        'Sensitivity-Weighted':  [metrics['Sensitivity-Weighted']],
        'Specificity-Weighted':  [metrics['Specificity-Weighted']],
        'Precision-Weighted':    [metrics['Precision-Weighted']],
        'F1-Score-Weighted':     [metrics['F1-Score-Weighted']],
    })


def get_tp_tn_fp_fn(original_labels, given_labels):
    TP = TN = FP = FN = 0
    for gt, pred in zip(original_labels, given_labels):
        if pred not in {0, 1}:
            pred = 0
        if   gt == 1 and pred == 1: TP += 1
        elif gt == 0 and pred == 0: TN += 1
        elif gt == 0 and pred == 1: FP += 1
        elif gt == 1 and pred == 0: FN += 1
    return TP, TN, FP, FN


averaging_technique = 'weighted'


def process_metrics_averaging(original_labels, given_labels):
    indexes_to_remove = {i for i, v in enumerate(given_labels) if v == 404}
    original_labels   = [v for i, v in enumerate(original_labels) if i not in indexes_to_remove]
    given_labels      = [v for i, v in enumerate(given_labels)    if i not in indexes_to_remove]

    precision    = round(precision_score(original_labels, given_labels, average=averaging_technique, zero_division=0), 3)
    sensitivity  = round(recall_score(original_labels,   given_labels, average=averaging_technique, zero_division=0), 3)
    f1           = round(f1_score(original_labels,       given_labels, average=averaging_technique, zero_division=0), 3)

    try:
        tn, fp, fn, tp = get_tp_tn_fp_fn(original_labels, given_labels)
    except:
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

    n0 = np.sum(np.array(original_labels) == 0)
    n1 = np.sum(np.array(original_labels) == 1)
    n  = n0 + n1
    specificity = round((specificity_0 * n0 + specificity_1 * n1) / n, 3)

    return sensitivity, specificity, precision, f1


def process_metrics_non_averaging(TP, TN, FP, FN):
    sensitivity = round(TP / (TP + FN) if (TP + FN) != 0 else 0, 3)
    recall      = sensitivity
    specificity = round(TN / (TN + FP) if (TN + FP) != 0 else 0, 3)
    precision   = round(TP / (TP + FP) if (TP + FP) != 0 else 0, 3)
    f1_score    = round((2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0, 3)
    return sensitivity, specificity, precision, f1_score


def calculate_metrics(original_labels, given_labels):
    TP, TN, FP, FN = get_tp_tn_fp_fn(original_labels, given_labels)
    fpr, tpr, _    = roc_curve(original_labels, given_labels)

    sensitivity_w, specificity_w, precision_w, f1_w = process_metrics_averaging(original_labels, given_labels)
    sensitivity,   specificity,   precision,   f1   = process_metrics_non_averaging(TP, TN, FP, FN)

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'Precision': precision, 'F1-Score': f1,
        'FPR': fpr, 'TPR': tpr,
        'Sensitivity-Weighted': sensitivity_w,
        'Specificity-Weighted': specificity_w,
        'Precision-Weighted':   precision_w,
        'F1-Score-Weighted':    f1_w,
    }


def calculate_metrics_for_crosswalk(ground_truth, llm_response):
    chunk_size = 39
    for start in range(0, len(ground_truth), chunk_size):
        cnt = 1
        gt_chunk   = ground_truth[start:start + chunk_size]
        pred_chunk = llm_response[start:start + chunk_size]

        for gt, pred in zip(gt_chunk, pred_chunk):
            if gt == 0 and pred == 1:
                with open('results/crosswalk_fp.txt', 'a') as f:
                    f.write(f"QUESTION-{cnt}\n")
            elif gt == 1 and pred == 0:
                with open('results/crosswalk_fn.txt', 'a') as f:
                    f.write(f"QUESTION-{cnt}\n")
            cnt += 1

    TP = TN = FP = FN = 0
    for gt, pred in zip(ground_truth, llm_response):
        if   gt == 1 and pred == 1: TP += 1
        elif gt == 0 and pred == 0: TN += 1
        elif gt == 0 and pred == 1: FP += 1
        elif gt == 1 and pred == 0: FN += 1

    fpr, tpr, _ = roc_curve(ground_truth, llm_response)

    sensitivity_w, specificity_w, precision_w, f1_w = process_metrics_averaging(ground_truth, llm_response)
    sensitivity,   specificity,   precision,   f1   = process_metrics_non_averaging(TP, TN, FP, FN)

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'Precision': precision, 'F1-Score': f1,
        'FPR': fpr, 'TPR': tpr,
        'Sensitivity-Weighted': sensitivity_w,
        'Specificity-Weighted': specificity_w,
        'Precision-Weighted':   precision_w,
        'F1-Score-Weighted':    f1_w,
    }


# --------------------------------------------------------------------------- #
#  DUMMY RESPONSE (for offline testing)                                        #
# --------------------------------------------------------------------------- #

dummy_response = """{
  "recommendations": [
    {
      "recommendation": "Adopt lightweight contribution interventions such as structured peer review assignments.",
      "positive_impact": "Increases the volume and quality of student contributions to OSS projects.",
      "evidence": "Over four years, projects with structured interventions received 3x more pull requests than control groups.",
      "confidence": 0.92
    }
  ]
}"""