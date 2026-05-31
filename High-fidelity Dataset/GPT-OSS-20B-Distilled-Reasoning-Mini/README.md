---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- text-generation
pretty_name: GPT-OSS-20B Distilled Reasoning Dataset Mini
size_categories:
- 1K<n<10K
language:
- en
---

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/yuc6SULgg2DNRDJQrGB5l.jpeg)

# Dataset Card for GPT-OSS-20B Distilled Reasoning Dataset Mini
GPT-OSS-20B Distilled Reasoning Dataset Mini
* (Multi-stage Evaluative Refinement Method for Reasoning Generation)
## Dataset Details and Description
This is a high-quality instruction fine-tuning dataset constructed through knowledge distillation, featuring detailed Chain-of-Thought (CoT) reasoning processes. The dataset is designed to enhance the capabilities of smaller language models in complex reasoning, logical analysis, and instruction following.
I generated this dataset through a meticulously designed three-stage process:
- **Seed Question Aggregation:** I extensively gathered and filtered seed questions from multiple high‑quality data sources, including self‑generated ones.

- **Knowledge Distillation & Cleaning:** I used `openai/gpt-oss-20b` (High) as the teacher model to generate in‑depth reasoning and responses for the seed questions, which then underwent a rigorous automated cleaning process.

- **LLM‑as‑a‑Judge Quality Assessment:** I employed `Qwen/Qwen3-235B-A22B-Instruct-2507` as the judge model to score the generated data across multiple dimensions, ensuring the quality of the final output.

- **Curated by:** [Jack Rong]
- **Funded by :** [Jack Rong]
- **Shared by :** [Jack Rong]
- **Language(s) (NLP):** [En]
- **License:** [apache-2.0]

- **Teacher Model**: `unsloth/gpt-oss-20b`
- **Judge Model**: `Qwen/Qwen3-235B-A22B-Instruct-2507`


#### Source Data
<div style="display: flex; gap: 10px; justify-content: space-around;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/hEWXvzYitkXka19sBs9k_.png" style="width: calc(33.33% - 10px);">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/61KVV5AJeDrm5FpMrxZzC.png" style="width: calc(33.33% - 10px);">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/YQ49_Kdu8kPY9VfWnAD0_.png" style="width: calc(33.33% - 10px);">
  </div>

### Phase 1: Seed Question Aggregation  
To ensure diversity and comprehensiveness in the distilled instruction sources, I employed a **multi-source aggregation strategy** to construct the seed question pool:  
<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/bm5_WgDdPTFnWFgGSBGTs.png" width="300">
- **Self-generated** (300 entries):  
  Using the **unsloth/Qwen3-4B-Instruct-2507** model, I generated **300 high-quality, representative initial questions** based on 7 predefined core thinking templates:  
  - Logical reasoning  
  - Basic mathematics  
  - Scientific knowledge  
  - Programming thinking  
  - Common sense  
  - Creative imagination  
  - Ethical dilemmas  

- **External Dataset Integration**:  
  To further enhance coverage, we sampled from multiple **established public datasets**, including:  
  - `facebook/natural_reasoning`  
  - `vicgalle/alpaca-gpt4`  
  - `openai/gsm8k`  
  - `fka/awesome-chatgpt-prompts`  
  - `microsoft/rStar-Coder`  
  - `nvidia/OpenScience`  

Prior to distillation, all seed questions underwent rigorous automated cleaning to filter entries with:  
- Missing referenced attachments  
- Non-English content  
- Excessively short queries  

This process resulted in **1,990 high-quality questions** for final distillation (from an initial pool of 1,994 entries).

### Phase 2: Distillation & Cleaning  

<div style="display: flex; gap: 10px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/P4LYLG5K6FNJ-xrHkFO0X.png" alt="Image 1" width="300">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/0kgR7OMbtEF7SQJ1Z-T7w.png" alt="Image 2" width="300">
</div>

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/knsldQhMFoy4eyF-X3qi-.png) 

This phase forms the core of generating high-quality chain-of-thought (CoT) content. The entire workflow ensures that **both inputs and outputs undergo rigorous quality verification** – not merely cleaning the inputs, but systematically validating the outputs.  

- **Distillation Generation**:  
  I employed `openai/gpt-oss-20b(High)` as the teacher model. For each seed question, a specially designed prompt template was used to explicitly instruct the model to:  
  (1) Conduct detailed **"Think step-by-step" chain-of-thought reasoning**, followed by  
  (2) Delivering the final answer.  
  This structured approach guarantees comprehensive reasoning before conclusion.  

- **Post-Distillation Cleaning**:  
  Raw CoT outputs and answers from the teacher model were **not directly adopted**. Instead, we implemented an additional quality assessment and cleaning round to:  
  - Eliminate low-quality outputs caused by **model hallucinations**, **logical inconsistencies**, or **formatting errors**  
  - Filter entries failing factual coherence or structural integrity checks  
  This critical step ensures **every sample** in the final dataset contains **high-quality, reliable reasoning processes and answers**.

### Phase 3: LLM-as-a-Judge Quality Scoring  
To quantitatively evaluate distilled data quality, we implemented an **LLM-as-a-Judge** assessment framework that measures multiple reasoning dimensions.  

- **Judge Model**:  
  We utilized the more powerful `Qwen/Qwen3-235B-A22B-Instruct-2507` model as the judge (235B parameters specification maintained for exactness per OpenAI/Anthropic model documentation conventions).  

- **Multi-Dimensional Evaluation Rubric**:  
  A comprehensive **three-aspect scoring system** was developed, requiring judge model to rate each entry on a 1-10 scale:  
  - **Input Score**: Measures question clarity, answerability, reasoning density, and contextual relevance  
  - **CoT Score**: Evaluates logical coherence, presence of **logical leaps**, constraint adherence, and alignment with final answer  
  - **Answer Score**: Assesses accuracy, completeness, **format adherence**, and robustness  

- **Weighted Scoring Formula**:  
  To emphasize reasoning and answer quality, we applied a weighted average:  
  `Total Score = 0.15 × Input Score + 0.35 × CoT Score + 0.5 × Answer Score`  

- **Scoring Results**:  
  After evaluating **1,946 entries** (excluding 44 failed samples), the dataset demonstrates **exceptional quality**, with most scores concentrated in the 8.0-10.0 range. This distribution statistically validates the effectiveness of our three-stage distillation process.
<div style="display: flex; gap: 12px; align-items: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/-5qm3NXBKM2q_Twg2DfHy.png"
       alt="图1"
       width="450"
       style="object-fit: contain;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/W7G5ua95zMGVTyaGxbq9N.png"
       alt="图2"
       width="280"
       style="object-fit: contain;">
</div>

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/_-c1UaAqYGn_2d81CPVtx.png"
     alt="800px 图像"
     width="650"
     style="display:block; margin:auto;">

### Direct Use

This dataset is primarily designed for instruction fine-tuning of language models, specifically to enhance their chain-of-thought (CoT) generation and complex reasoning capabilities. It applies to the following scenarios:  
- **Model Fine-Tuning**: Serves as high-quality training data for fine-tuning language models of various scales to improve their logical reasoning, step-by-step problem decomposition, and instruction-following abilities in English contexts.  
- **Academic Research**: Enables investigation into knowledge distillation efficacy, CoT generation mechanisms, and the validity of LLM-as-a-Judge evaluation methodologies.  
- **Model Evaluation**: Functions as a benchmark to assess model performance on complex reasoning tasks.  

### Out-of-Scope Use
- **Critical Fact-Checking in Production**: Generated by AI models, this dataset may contain factual inaccuracies. It **shall not** be used in production environments requiring high factual precision (e.g., medical, legal, or financial advice) without rigorous fact verification.  
- **Non-English Tasks**: Content is exclusively in English and **is not suitable** for direct fine-tuning of Chinese or other language models.  
- **Malicious Applications**: **Strictly prohibited** for any harmful purposes, including but not limited to generating toxic, unethical, biased, or misleading content.

### Dataset Structure
The dataset is provided in `.jsonl` format, where each line constitutes a JSON object. No predefined train/validation/test splits are included; users are advised to perform random splits according to their specific requirements.

### Data Fields
- **id (string)**: Unique identifier for the sample.  
- **category (string)**: Category to which the question belongs.  
- **input (string)**: User's instruction or query.  
- **CoT_content (string)**: Chain-of-thought reasoning process generated by the teacher model.  
- **answer_content (string)**: Final answer generated by the teacher model.  
- **judge_reviews (list)**: List containing review comments from judge models.  
- **scores (dict)**: Dictionary containing specific scores for `input`, `cot`, and `answer` dimensions.  
- **total_score (float)**: Final quality score calculated using a weighted formula.  
- **num_judgments (int)**: Number of judgments (defaulting to 1).  

### Data Instance
```json
{"id": "hf_openai_gsm8k_00173", "category": "gsm8k",
 "input": "Mitchell has 30 pencils. He has 6 more pencils than Antonio. How many pencils does Mitchell and Antonio have together?",
 "CoT_content": "<|channel|>analysis<|message|>We need to parse the problem: Mitchell has 30 pencils. He has 6 more pencils than Antonio. So Antonio's pencils = Mitchell - 6 = 30 - 6 = 24. Then together they have 30 + 24 = 54 pencils.\n\nCheck: If Mitchell has 30 and Antonio has 24, difference is indeed 6. So answer 54.\n\nThus final answer: 54 pencils.\n\nWe need to produce answer with \"Final answer:\" on a new line.",
"answer_content": "To find the total number of pencils:\n\n1. **Determine Antonio's pencils**  \n   Mitchell has 6 more than Antonio.  \n   \\[\n   \\text{Antonio} = \\text{Mitchell} - 6 = 30 - 6 = 24\n   \\]\n\n2. **Add their pencils together**  \n   \\[\n   \\text{Total} = \\text{Mitchell} + \\text{Antonio} = 30 + 24 = 54\n   \\]\n\nSo, Mitchell and Antonio have a total of **54 pencils**.\n\nFinal answer: 54",
"judge_reviews": [{"id": "hf_openai_gsm8k_00173", "input_review": "The input is clear, directly stated, and requires basic arithmetic reasoning with a well-defined solution path.", "cot_review": "The chain-of-thought is logically coherent, correctly interprets the relationship between Mitchell and Antonio's pencils, and verifies the result.", "answer_review": "The final answer is correct, clearly formatted, includes proper justification, and ends with 'Final answer: 54' as required.",
"scores": {"input": 10, "cot": 10, "answer": 10}}], "scores": {"input": 10, "cot": 10, "answer": 10},
"total_score": 10.0, "num_judgments": 1}
```


### Data Splits  
This dataset **does not include predefined train/validation/test splits**. Users are **advised to perform random splits** according to their specific requirements to:  
- Prevent data leakage during evaluation  
- Maintain category distribution balance (critical for reasoning tasks)  
- Align with downstream task constraints 
# Acknowledgments
The creation of this dataset was made possible by the following outstanding open-source models and datasets. We extend our sincere gratitude:

- **Teacher Model**: `openai/gpt-oss-20b`  
- **Judge Model**: `Qwen/Qwen3-235B-A22B-Instruct-2507`  
- **Seed Question Sources**:  
  `facebook/natural_reasoning`,  
  `vicgalle/alpaca-gpt4`,  
  `openai/gsm8k`,  
  `fka/awesome-chatgpt-prompts`,  
  `microsoft/rStar-Coder`,  
  `nvidia/OpenScience`



```bibtex
@dataset{jackrong2025gptossdistilled,
  author       = {JACKRONG},
  title        = {Jackrong/GPT-OSS-20B-Distilled-Reasoning-Mini},
  month        = aug,
  year         = {2025},
  publisher    = {Hugging Face},
  version      = {1.0.0},
  url          = {https://huggingface.co/datasets/Jackrong/GPT-OSS-20B-Distilled-Reasoning-Mini}
}
```
