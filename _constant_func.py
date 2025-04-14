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


# You have been given the full text of the article titled [A Four-Year Study of Student Contributions to OSS vs. OSS4SG with a Lightweight Intervention]. Your task is to extract actionable recommendations from the article. An actionable recommendation is a practical, evidence-based suggestion that provides specific, clear steps or instructions which, when implemented, are expected to produce tangible and positive results. Adopting these actionable recommendations can help make open-source software projects more sustainable. Let's break down the problem into the following steps: Step 1: Carefully read each sentence of the article and identify recommendations. Look for imperative sentences or phrases that give commands, make requests, or offer instructions to direct or persuade someone to perform a specific action. IF NO SUCH RECOMMENDATIONS CAN BE FOUND, THEN PRINT “NO ACTIONABLE CAN BE DERIVED”, AND STOP PROCESSING HERE [DO NOT MOVE TO NEXT STEP] Step 2: For each recommendation, identify the positive impacts mentioned in the article that would result from its adoption. If no such impact can be found for a particular recommendation, state "NO IMPACT FOUND". Step 3:  For each recommendation, provide empirical evidence, presented in the article, that supports the claim that implementing the proposed action will result in the anticipated outcome. If no evidence of impact can be found for a particular recommendation, state "NO EVIDENCE FOUND". Step 4: For each recommendation in the final list, assign a confidence level on a scale of 0 to 1, indicating how confident you are extracting that actionable. STEP 5: Provide the FINAL LIST of recommendations. For each of the recommendations, include: “POSITIVE IMPACT”, “EVIDENCE”, and “CONFIDENCE”. Your attention to detail in this task is crucial, as its successful completion holds significant importance for my career advancement.



prompt_template = """
You are a proficient literature reviewer. Given the full text of an article, your task is to extract actionable recommendations in a structured JSON format.
Full Text of the ARTICLE:
"""

prompt_template_for_assigning_categories ="""
You are a prolific literature reviewer analyzing practical suggestions—called actionables—to enhance the sustainability of Open Source Software (OSS) projects. You are provided with the following:

i) A list of actionables (i.e., practical, targeted suggestions for improving OSS project sustainability),

ii) A fixed set of Actionable Categories, each with a clear Category Definition and Criteria for Assignment.

Your task is to assign the most appropriate criterion (from the predefined list under each category) to each actionable. Approach the task step by step.

Instructions:
1) Read the actionable carefully.

2) Identify the category it fits best into by using the Category Definition.

3) Match the actionable to one specific criterion under that category by reasoning through the Criteria for Assignment.

4) Output only the criterion name that best matches the actionable. Do not output the the reasoning, or any explanatory text.

5) If multiple criteria seem plausible, choose the most specific one.

ReACT Categories and Criteria:
\section{Definition of ReACT Categories}
\subsection{New Contributor Onboarding and Involvement}
\textit{Definition}. This category focuses on ensuring that new contributors can easily join, understand, and meaningfully contribute to the project.  

\textit{Criteria for Assignment}: \textit{a)} Actionable facilitates the integration of new contributors by providing mentorship, onboarding materials, or simplifying the contribution process; \textit{b)} Actionable relates to improving project documentation or offering better support mechanisms for first-time contributors; \textit{c)} Actionable helps build a welcoming, inclusive, and open culture for new participants.


\subsection{Code Standards and Maintainability}
\textit{Definition}. This category deals with ensuring that the codebase adheres to established standards, making it easier to maintain and scale. It includes efforts to ensure code readability, modularity, and compliance with coding best practices.

\textit{Criteria for Assignment}. \textit{a)} Actionable relates to improving the quality, readability, or structure of the codebase; \textit{b)} Actionable includes efforts to enforce coding guidelines, refactor code for better maintainability, or reduce technical debt; Actionable includes the use of linters, formatters, or static code analysis tools.


\subsection{Automated Testing and Quality Assurance}
\textit{Definition}. This category focuses on ensuring the project’s robustness and reliability through automated testing practices, such as unit, integration, and end-to-end tests. It also includes broader quality assurance activities.

\textit{Criteria for Assignment}. \textit{a)} Actionable involves the implementation or improvement of automated testing frameworks and testing strategies; \textit{b)} Actionable includes practices that ensure the detection of bugs early in the development cycle and ensure high-quality releases; 


\subsection{Community Collaboration and Engagement}
\textit{Definition}. This category deals with activities that foster collaboration, communication, and engagement within the OSS community. It includes practices for keeping the community active and involved.

\textit{Criteria for Assignment}. \textit{a)} Actionable aims to improve communication between contributors, maintainers, and users; Actionable involves organizing community-driven events, discussions, or collaborations, as well as platforms to enhance transparency and teamwork; Actionable relates to tools and processes for better community governance and decision-making.

\subsection{Documentation Practices}
\textit{Definition}. This category focuses on ensuring that the project’s documentation is thorough, up-to-date, and easily accessible. Documentation practices are crucial for both current and future contributors.

\textit{Criteria for Assignment}. \textit{a)} Actionable focuses on improving the quality, clarity, or accessibility of project documentation, such as user guides, API references, or contributor guides; \textit{b)} Actionable includes practices for keeping documentation synchronized with the codebase and ensuring it meets the needs of different stakeholders; \textit{c)} Actionable involves translation efforts or making documentation more accessible to non-expert audiences.

\subsection{Project Management and Governance}
\textit{Definition}. This category deals with the governance structure and project management practices that keep the project organized, transparent, and sustainable over the long term. 

\textit{Criteria for Assignment}. \textit{a)} Actionable enhances the governance model, clarifies roles and responsibilities, or improves the decision-making process; \textit{b)} Actionable involves defining or refining processes for issue triaging, release management, or conflict resolution; \textit{c)} Actionable includes efforts to improve the transparency of project goals, progress, and decision-making.


\subsection{Security Best Practices and Legal Compliance}
\textit{Definition}. This category addresses efforts to secure the project and ensure compliance with relevant legal standards, such as licenses, data privacy laws, and security protocols.

\textit{Criteria for Assignment}. \textit{a)} Actionable focuses on improving the security posture of the project by following best practices, addressing vulnerabilities, or conducting audits; \textit{b)} Actionable involves ensuring compliance with open-source licenses, setting up contributor license agreements (CLAs), or aligning with data privacy regulations; \textit{c)} Actionable includes security measures such as dependency management, security audits, and secure coding practices.

\subsection{CI/CD and DevOps Automation}
\textit{Definition}. This category deals with continuous integration and continuous deployment (CI/CD) processes that automate building, testing, and deployment pipelines. It also includes broader DevOps automation tasks.

\textit{Criteria for Assignment}. \textit{a)} Actionable involves the setup or enhancement of CI/CD pipelines to ensure faster, reliable, and automated releases; \textit{b)} Actionable relates to automating infrastructure provisioning, containerization, or deployment to cloud environments; \textit{c)} Actionable includes the integration of DevOps practices that ensure smooth, automated, and repeatable processes for software development, testing, and deployment.

ACTIONABLE:
"""



CoT_template = """
### **Definition of Actionable Recommendation:**

An actionable recommendation is a **practical, evidence-based suggestion** that provides **specific, clear steps or instructions** which, when implemented, are expected to produce **tangible and positive results**. These recommendations should **contribute to making open-source software (OSS) projects more sustainable**.

### **Extraction Process:**

#### **Step 1: Identify Actionable Recommendations**

- Carefully read each sentence of the article.
- Identify **recommendations** by looking for **imperative sentences or phrases** that give commands, make requests, or offer instructions.
- If **no actionable recommendations** are found, return the following JSON object and **stop processing**:

```json
{
  "message": "NO ACTIONABLE CAN BE DERIVED"
}
```

#### **Step 2: Identify Positive Impacts**

- For each recommendation, extract the **positive impacts** mentioned in the article that would result from adopting it.
- If **no impact** is explicitly mentioned, assign the value `"NO IMPACT FOUND"`.

#### **Step 3: Identify Supporting Evidence**

- Extract **empirical evidence** from the article that supports the claim that implementing the recommendation leads to the stated impact.
- If **no evidence** is found, assign the value `"NO EVIDENCE FOUND"`.

#### **Step 4: Assign Confidence Levels**

- For each recommendation, assign a **confidence level (0 to 1)** based on how clearly and strongly the article supports it.

#### **Step 5: Output the Final List in JSON Format**

- Present the final list of recommendations in the following structured JSON format:

```json
{
  "recommendations": [
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    },
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    }
  ]
}
```

Your precision in extracting and structuring the data is **critical**, as this task holds significant importance for my career advancement.

"""



RA_template = """
For each actionable recommendation, follow these reasoning steps:

1. **Reasoning**: Determine whether a specific recommendation can be derived from the text. A recommendation is any directive, suggestion, or imperative sentence that offers practical guidance on improving OSS project sustainability.
   - **Action**: If a recommendation is found, proceed to the next step. If **no actionable recommendations** can be derived, return the following JSON object and **stop processing**:
   
   ```json
   {
     "message": "NO ACTIONABLE CAN BE DERIVED"
   }
   ```

2. **Reasoning**: Identify the potential **positive impacts** of adopting each recommendation, based on the article's content. This could include improvements in project sustainability, contribution rates, community engagement, or other measurable outcomes.
   - **Action**: For each recommendation, extract the positive impacts or indicate `"NO IMPACT FOUND"` if no impact is explicitly mentioned.

3. **Reasoning**: Provide **empirical evidence** from the article that supports the claim that adopting the recommendation will result in the stated positive impact.
   - **Action**: If evidence is present, provide it; otherwise, return `"NO EVIDENCE FOUND"`.

4. **Reasoning**: Evaluate the strength of the recommendation, its impact, and the evidence supporting it.
   - **Action**: Assign a **confidence level (0 to 1)** based on how strongly the article supports the recommendation.

5. **Final Action**: Provide the recommendations in a structured JSON format with the following fields:

```json
{
  "recommendations": [
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    },
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    }
  ]
}
```
"""






IP_template = """
Please present your findings in a structured **JSON format** with the following information for each recommendation:

- **recommendation**: The actionable recommendation extracted from the article.
- **positive_impact**: The expected positive impact of the recommendation (if mentioned in the article).
- **evidence**: Any supporting evidence presented in the article for the recommendation.
- **confidence**: A confidence score between 0 and 1, indicating how confident you are in the extracted recommendation.

Output format:

```json
{
  "recommendations": [
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    },
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": <confidence_score (0 to 1)>
    }
  ]
}
```
"""



allowable_models = ["meta.llama3-1-405b-instruct-v1:0", "anthropic.claude-3-5-haiku-20241022-v1:0", 
                    "mistral.mistral-large-2407-v1:0", "anthropic.claude-3-opus-20240229-v1:0",
                      "anthropic.claude-v2", "meta.llama3-1-70b-instruct-v1:0", "deepseek-r1:1.5b",
                      "llama3.2:3b-instruct-q4_K_M", "mixtral:8x7b-instruct-v0.1-q4_K_M", "qordmlwls/llama3.1-medical:latest",
                      "medllama2:latest", "meditron:70b", "llama3.2:latest", "anthropic.claude-3-7-sonnet-20250219-v1:0", 
                      "anthropic.claude-3-5-sonnet-20241022-v2:0", "tinyllama", "llama3.3:70b", "qwen2.5:72b",
                        "deepseek-r1:7b", "thewindmom/llama3-med42-8b:latest", "mixtral:latest"]

allowable_prompting_methods = ["IP", "CoT", "RA"]




# bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

def remove_newlines(text):
    return text.replace("\n", "").replace("\r", "")


def clean_response(response):
    json_start_index = response.find("{")
    json_end_index = response.rfind("}")

    # Step 2: Extract the JSON part by slicing the response
    json_text = response[json_start_index:json_end_index + 1]

    json_text = re.sub(r':\s*<([^>]+)>', r': "\1"', json_text)  # Replace <...>
    # print(json_text)

    # Step 3: Remove any extra non-JSON characters that might interfere with the parsing
    # You can clean the string by ensuring that the JSON structure is correctly formatted.
    json_text = re.sub(r'(?<!\\)("[^"]*")\s*:\s*(?=\S)', r'\1: ', json_text)  # Remove extra spaces after colons

    # Step 4: Use regular expression to validate and extract the JSON content
    json_part = re.search(r'\{.*\}', json_text, re.DOTALL)

    parsed_json=None

    # Step 5: Check if a JSON part was found and parse it
    if json_part:
        try:
            extracted_json = json_part.group()  # Get the full JSON content
            parsed_json = json.loads(extracted_json)  # Convert the JSON string into a Python dictionary
            print(parsed_json)
        except:
            print(f"Error decoding JSON:")
    else:
        parsed_json = response
        print("No valid JSON found in the response.")

    if(parsed_json==None):
        parsed_json = {'message': 'NO ACTIONABLE CAN BE DERIVED'}
    return parsed_json





def get_pdfs(local_dir):
    pdf_files = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_article_title(markdown_text):
    """Extracts all level 1 headings (H1) from a Markdown string."""
    return re.findall(r'^# (.+)', markdown_text, re.MULTILINE)

def clean_article_text(text):
    """Removes everything after '### ACKNOWLEDGMENTS' or '### REFERENCES' (case-insensitive)."""
    pattern = r'(?i)### (ACKNOWLEDGMENTS|REFERENCES|ACKNOWLEDGMENT|REFERENCE).*'  # Case-insensitive match
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()

def extract_article_link(text):
    """Extracts the DOI or article link from the given text."""
    url_pattern = r'https?://[^\s\)]+'
    return re.findall(url_pattern, text)






# Define the expected JSON schema using Pydantic
class ClassificationResponse(BaseModel):
    reason_for_the_label: str
    label: int  # Assuming 'Label' is an integer

def extract_first_binary(var_name):
    match = re.search(r'[01]', var_name)  # Find the first occurrence of '0' or '1'
    return match.group(0) if match else 0


def fix_json(json_input, response):
    """
    Ensures the input is a JSON string or a dictionary and always returns a dictionary.
    If input is a dictionary, return it as-is.
    If input is a valid JSON string, return parsed JSON as a dictionary.
    If input is an invalid JSON string, attempts to fix it by trimming trailing characters.
    """
    # If input is already a dictionary, return it directly
    if isinstance(json_input, dict):
        return json_input  

    # Ensure input is a string (or bytes), otherwise return error JSON
    if not isinstance(json_input, (str, bytes, bytearray)):
        return {"reason_for_the_label": response, "label": extract_first_binary(response)}

    # First, check if the JSON is already valid
    try:
        parsed_json = json.loads(json_input)
        if isinstance(parsed_json, dict):
            return parsed_json  # Ensure it's a dictionary
        else:
            return {"reason_for_the_label": response, "label": extract_first_binary(response)}
    except json.JSONDecodeError:
        pass  # If invalid, proceed with fixing

    # Try trimming trailing characters
    for i in range(len(json_input), 0, -1):  
        try:
            parsed_json = json.loads(json_input[:i])  # Try parsing progressively shorter substrings
            if isinstance(parsed_json, dict):
                return parsed_json  # Ensure it's a dictionary
        except json.JSONDecodeError:
            continue  # Keep trimming

    # If all attempts fail, return error JSON
    return {"reason_for_the_label": response, "label": extract_first_binary(response)}


def get_dataframes(metrics, modality):
    return pd.DataFrame({
            'Modality': [modality],  
            'TP': [metrics['TP']], 
            'TN': [metrics['TN']], 
            'FP': [metrics['FP']], 
            'FN': [metrics['FN']], 
            'Sensitivity': [metrics['Sensitivity']], 
            'Specificity': [metrics['Specificity']],  
            'Precision': [metrics['Precision']],  
            'F1-Score': [metrics['F1-Score']],
            'FPR': [metrics['FPR']],
            'TPR': [metrics['TPR']],
            'Sensitivity-Weighted': [metrics['Sensitivity-Weighted']],
            'Specificity-Weighted': [metrics['Specificity-Weighted']],
            'Precision-Weighted': [metrics['Precision-Weighted']],
            'F1-Score-Weighted': [metrics['F1-Score-Weighted']],
            
        })
def get_tp_tn_fp_fn(original_labels, given_labels):  
    TP = TN = FP = FN = 0
    for gt, pred in zip(original_labels, given_labels):
        if pred not in {0,1}:
            pred = 0
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
    return TP, TN, FP, FN

######### CODE FOR EVALUATION ####################
averaging_technique = 'weighted' # It is either macro or weighted

def process_metrics_averaging(original_labels, given_labels): #    
    indexes_to_remove = {i for i, value in enumerate(given_labels) if value == 404}
    original_labels = [value for i, value in enumerate(original_labels) if i not in indexes_to_remove]
    given_labels = [value for i, value in enumerate(given_labels) if i not in indexes_to_remove]


    precision = round(precision_score(original_labels, given_labels, average=averaging_technique, zero_division=0), 3)
    sensitivity = round(recall_score(original_labels, given_labels, average=averaging_technique,zero_division=0), 3) #equivalent to recall
    f1 = round(f1_score(original_labels, given_labels, average=averaging_technique,zero_division=0), 3)

    try:
        tn, fp, fn, tp = get_tp_tn_fp_fn(original_labels, given_labels)
    except:
        tn, fp, fn, tp = 0,0,0,0
    
    specificity_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_1 = tp / (tp + fn) if (tp + fn) > 0 else 0


    # Compute class proportions
    n0 = np.sum(np.array(original_labels) == 0)
    n1 = np.sum(np.array(original_labels) == 1)
    n = n0 + n1

    # Weighted specificity
    specificity = round((specificity_0 * n0 + specificity_1 * n1) / n, 3)

    return sensitivity, specificity, precision, f1


def process_metrics_non_averaging(TP, TN, FP, FN): #This calculates for only one class (Positive)    
    sensitivity = round(TP / (TP + FN) if (TP + FN) != 0 else 0, 3) #equivalent to recall
    recall = sensitivity

    specificity = round(TN / (TN + FP) if (TN + FP) != 0 else 0, 3)

    precision = round(TP / (TP + FP) if (TP + FP) != 0 else 0, 3)

    f1_score = round((2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0, 3)
    
    return sensitivity, specificity, precision, f1_score


def calculate_metrics(original_labels, given_labels):
    TP, TN, FP, FN = get_tp_tn_fp_fn(original_labels, given_labels)
    fpr, tpr, thresholds = roc_curve(original_labels, given_labels)

    sensitivity_weighted, specificity_weighted, precision_weighted, f1_weighted = process_metrics_averaging(original_labels, given_labels)    
    sensitivity, specificity, precision, f1 = process_metrics_non_averaging(TP, TN, FP, FN)

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'FPR': fpr,
        'TPR': tpr,
        'Sensitivity-Weighted': sensitivity_weighted,
        'Specificity-Weighted': specificity_weighted,
        'Precision-Weighted': precision_weighted,
        'F1-Score-Weighted': f1_weighted,
    }


def calculate_metrics_for_crosswalk(ground_truth, llm_response):    
    # Loop through the ground truth and LLM responses for identifying the FP and FN cases
    chunk_size = 39
    for start in range(0, len(ground_truth), chunk_size):
        cnt = 1  # Question counter

        gt_chunk = ground_truth[start:start + chunk_size]
        pred_chunk = llm_response[start:start + chunk_size]

        for gt, pred in zip(gt_chunk, pred_chunk):
            if gt == 0 and pred == 1:
                with open('results/crosswalk_fp.txt', 'a') as file:
                    file.write(f"QUESTION-{cnt}\n")
            elif gt == 1 and pred == 0:
                with open('results/crosswalk_fn.txt', 'a') as file:
                    file.write(f"QUESTION-{cnt}\n")
            cnt += 1  # Increment question counter
    # Loop through the ground truth and LLM responses for identifying the FP and FN cases

    # Loop through the ground truth and LLM responses for calculating the metrics
    TP = TN = FP = FN = 0
    cnt=1
    for gt, pred in zip(ground_truth, llm_response):
        if gt == 1 and pred == 1:
            TP += 1  # True Positive
        elif gt == 0 and pred == 0:
            TN += 1  # True Negative
        elif gt == 0 and pred == 1:
            FP += 1  # False Positive
        elif gt == 1 and pred == 0:
            FN += 1  # False Negative
        cnt=cnt+1
    
    fpr, tpr, thresholds = roc_curve(ground_truth, llm_response)

    sensitivity_weighted, specificity_weighted, precision_weighted, f1_weighted = process_metrics_averaging(ground_truth, llm_response)    
    sensitivity, specificity, precision, f1 = process_metrics_non_averaging(TP, TN, FP, FN)

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'FPR': fpr,
        'TPR': tpr,
        'Sensitivity-Weighted': sensitivity_weighted,
        'Specificity-Weighted': specificity_weighted,
        'Precision-Weighted': precision_weighted,
        'F1-Score-Weighted': f1_weighted,
    }

######### CODE FOR EVALUATION ####################



dummy_response = """Please present your findings in a structured **JSON format** with the following information for each recommendation:

- **recommendation**: The actionable recommendation extracted from the article.
- **positive_impact**: The expected positive impact of the recommendation (if mentioned in the article).
- **evidence**: Any supporting evidence presented in the article for the recommendation.
- **confidence**: A confidence score between 0 and 1, indicating how confident you are in the extracted recommendation.

Output format:

{
  "recommendations": [
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": "<confidence_score (0 to 1)>"
    },
    {
      "recommendation": "<extracted recommendation>",
      "positive_impact": "<extracted impact or 'NO IMPACT FOUND'>",
      "evidence": "<extracted evidence or 'NO EVIDENCE FOUND'>",
      "confidence": "<confidence_score (0 to 1)>"
    }
  ]
}


skvmdksv
ksdmvksdv

smdvlsdmvlk"""