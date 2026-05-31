---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- table-question-answering
- summarization
- translation
- text-generation
language:
- zh
pretty_name: Chinese-Qwen3-235B-Thinking-2507-Distill-100k
size_categories:
- 10K<n<100K


---
> 📌 Note: The English translation of this dataset card is provided below.
# Chinese-Qwen3-235B-Thinking-2507-Distill-100k

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/kEL1UykP-JcEzjFDwgA0Z.jpeg" width="1000" />

## Dataset Summary

**Chinese-Qwen3-235B-Thinking-2507-Distill-100k** 是一个包含约 **100k** 条高质量中文推理与指令数据的数据集，由 **Qwen-3-235B-A22B-Thinking-2507**（官方 *Thinking 模式*，上下文长度 32K）蒸馏生成。  

该数据集覆盖了多个重要领域：  
- **数学与工程任务**（Mathematics, Applied Math, Advanced Math）  
- **通用知识与写作**（General Knowledge, Language & Writing）  
- **技术与编程**（Technology & Programming）  
- **商业与经济**（Business & Economics）  
- **教育、医疗、生活与文化等其他场景**  

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/5yZrz_fae27T9_UKfRQV3.png" width="1000" />


数据特点是以复杂推理为主，强调 **多步思维链（Chain-of-Thought, CoT）** 的生成与保留，可广泛用于训练和评估中文推理模型。

在构建过程中：  
- **数据规模**：原始种子问题约 **110,000 条**，经清洗与筛选后保留约 **100,000 条**高质量样本。  
- **生成配置**：蒸馏过程中采用 Qwen3 官方推荐参数（**温度 = 0.6**），不额外添加系统提示词，以确保保留教师模型的原生推理风格与答案能力。  
- **清理步骤**：在蒸馏前后均进行了初步两轮清理，去除空样本、结构异常与冗余数据，保证了整体的一致性和可用性。  
- **模型环境**：所有数据由 **OpenRouter 官方路由满血版 Qwen-3-235B-A22B-Thinking-2507** 在 **32K 上下文窗口**下生成，能够支持长上下文任务。
- **供应商** ：Cerebras，novitaAI，Chutes

---

## Data Fields

每条样本包含以下字段：

- **Generator**: 教师模型名称（本数据集中为 `qwen-3-235b-a22b-thinking-2507`）  
- **Input**: 用户任务或问题（中文）  
- **CoT_content**: 教师模型的逐步推理链（思维过程）  
- **Answer_content**: 教师模型给出的最终答案或结论  
- **From**: 原始数据来源（如 `coig/neo`）  
- **question_len / anwser_len / reasoning_len**: 输入、答案、推理链的长度统计（便于数据分析与过滤）  

---

## Dataset Statistics & Visualization

本数据集附带全面的统计与可视化结果，帮助研究者快速理解数据的分布情况。  

### 1. Dataset Overview

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/uOtiz6akVHN_DOb5QSJf2.png" width="1000" />

- **总样本数**：106,950  
- **平均长度**：Question ≈ 56 字；Answer ≈ 1715 字；Reasoning ≈ 1267 字  
- **复杂度分布**：Complex (53.1%) 与 Very Complex (38.7%) 占比超过九成，反映出任务的高难度和深度。  
- **领域覆盖**：数学 (48,020)、通用知识 (15,934)、技术与编程 (12,042)、语言写作 (9,181)、商业经济 (7,766) 等。  

---

### 2. Content Length Analysis

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/iVZVzGaKR5gJXnIWWt6Co.png" width="1000" />

- **问题文本**：多为简短输入（≤100 字）。  
- **推理与答案**：具有明显长尾分布，部分样本推理链超过 5,000 字符。  
- **任务复杂度**：Very Complex 任务的平均字符数超过 4,100，显著高于 Simple/Moderate 类任务。  

---

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/8l9JqV_pxTwCvJH8nIzz3.png" width="1000" />

- 不同领域的任务推理链长度存在差异，其中数学、通用知识和编程领域的平均推理长度均超过 1,300 字。  
- 医疗、商业和教育等领域中，复杂与极复杂任务的比例显著偏高（60%+）。  
- 综合来看，该数据集在各个领域均提供了足够的难度层次，适合研究 **任务复杂度与推理链长度对模型性能的影响**。  

---

## Intended Uses

- **模型微调（SFT）**：作为中小规模模型的高难度训练语料，用于提升复杂推理与指令跟随能力。  
- **研究分析**：适用于对比研究 *思维链长度 vs. 模型表现* 的实验。  
- **多领域扩展**：涵盖数学、编程、通用写作等任务，可与其它语料混合使用，提升综合性能。  

---

## Limitations

- **领域覆盖有限**：尽管包含数学、编程和通用知识，但在法律、金融等垂直领域仍相对不足。  
- **输出长度问题**：部分思维链过长，可能导致小模型训练时出现冗长回答。  
- **潜在噪声**：教师模型的幻觉或语义冗余可能仍存在少量残留。  

建议用户在下游应用前，根据自身需求进一步清洗或裁剪!  

---

## License

- **Dataset License**: Apache-2.0  
- **Teacher Model License**: 遵循 Qwen 官方条款  

---

## How to Load

```python
from datasets import load_dataset

ds = load_dataset("Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k", split="train")
print(ds[0])
```

---

## Citation

如果你在研究或产品中使用了本数据集，请引用：  

```bibtex
@dataset{jackrong_zh_qwen3_thinking_distill_100k_2025,
  title   = {Chinese-Qwen3-235B-Thinking-2507-Distill-100k},
  author  = {Soren},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k},
  note    = {Chinese reasoning distillation dataset with Input, CoT_content, and Answer_content fields}
}
```

---

## Acknowledgements & References

本数据集的构建基于多个开源项目与研究成果。在此谨致谢意：  
- **Cong Liu et al.** – *Chinese dataset distilled from DeepSeek-R1-671b*  
- **m-a-p/neo_sft_phase2**  
- **hfl/stem_zh_instruction**
- **meta-math/GSM8K_zh**  
- **EduChat-Math**
- **Haijian/Advanced-Math**  
- **gavinluo/applied_math**    

> 本项目在上述开源数据的基础上，经过整理、清洗与二次蒸馏，最终形成 **Chinese-Qwen3-235B-Thinking-2507-Distill-100k**。衷心感谢所有贡献者与社区的支持。  

---

# Chinese-Qwen3-235B-Thinking-2507-Distill-100k

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/kEL1UykP-JcEzjFDwgA0Z.jpeg" width="1000" />

---

## Dataset Summary

**Chinese-Qwen3-235B-Thinking-2507-Distill-100k** is a dataset containing around **100k** high-quality Chinese reasoning and instruction-following samples, distilled from **Qwen-3-235B-A22B-Thinking-2507** **(official Thinking mode**, **32K context length**).  

---

The dataset spans multiple domains:  
- **Mathematics and Engineering tasks** (Mathematics, Applied Math, Advanced Math)  
- **General knowledge and writing** (General Knowledge, Language & Writing)  
- **Technology and Programming**  
- **Business and Economics**  
- **Education, Healthcare, Daily life, and Cultural scenarios**  

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/5yZrz_fae27T9_UKfRQV3.png" width="1000" />

The dataset emphasizes **complex multi-step reasoning** with explicit preservation of **Chain-of-Thought (CoT)**, making it highly suitable for training and evaluating Chinese reasoning models.

Construction details:  
- **Scale**: Initially ~110,000 seed questions. After cleaning and filtering, ~100,000 high-quality samples remain.  
- **Generation setup**: Distillation followed Qwen3 official recommended parameters (**temperature = 0.6**, **Top_P = 0.95**). No additional system prompts were used, ensuring the **native reasoning chains** and **answers** of the teacher model are faithfully preserved.  
- **Cleaning process**: Two rounds of pre- and post-distillation cleaning removed empty samples, malformed structures, and redundant entries, ensuring overall consistency and usability.  
- **Teacher environment**: All data was generated using **OpenRouter's full-powered Qwen-3-235B-A22B-Thinking-2507**, with a **32K context window**, enabling support for long-context tasks. 
- **Providers**: Cerebras, novitaAI, Chutes

---

## Data Fields

Each sample includes:  

- **Generator**: Name of the teacher model (here, `qwen-3-235b-a22b-thinking-2507`)  
- **Input**: User task or query (Chinese)  
- **CoT_content**: Teacher’s step-by-step reasoning chain (thought process)  
- **Answer_content**: Teacher’s final answer or conclusion  
- **From**: Source dataset (e.g., `coig/neo`)  
- **question_len / anwser_len / reasoning_len**: Length statistics for input, answer, and reasoning chain (useful for analysis and filtering)  

---

## Dataset Statistics & Visualization

The dataset comes with comprehensive statistics and visualization to help researchers quickly understand its distribution.  

### 1. Dataset Overview

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/uOtiz6akVHN_DOb5QSJf2.png" width="1000" />

- **Total samples**: 106,950  
- **Average length**: Question ≈ 56 chars; Answer ≈ 1715 chars; Reasoning ≈ 1267 chars  
- **Complexity distribution**: Complex (53.1%) and Very Complex (38.7%) together exceed 90%, showing the dataset is dominated by high-difficulty tasks.  
- **Domain coverage**: Mathematics (48,020), General Knowledge (15,934), Technology & Programming (12,042), Writing (9,181), Business & Economics (7,766), etc.  

---

### 2. Content Length Analysis

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/iVZVzGaKR5gJXnIWWt6Co.png" width="1000" />

- **Questions**: Mostly short inputs (≤100 chars).  
- **Reasoning & Answers**: Long-tailed distribution, with some reasoning chains exceeding 5,000 chars.  
- **Complexity**: Very Complex tasks average over 4,100 chars, much longer than Simple/Moderate tasks.  

---

### 3. Domain and Complexity Analysis

<img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/8l9JqV_pxTwCvJH8nIzz3.png" width="1000" />

- Reasoning chain lengths differ by domain: Mathematics, General Knowledge, and Programming average over 1,300 chars.  
- In Healthcare, Business, and Education, Complex and Very Complex tasks dominate (>60%).  
- Overall, the dataset provides sufficient depth across domains, making it suitable for studying **the relationship between task complexity, reasoning length, and model performance**.  

---

## Intended Uses

- **Supervised Fine-Tuning (SFT)**: High-difficulty training corpus for small- to mid-scale models, enhancing reasoning and instruction-following ability.  
- **Research**: Comparative studies on *reasoning length vs. model performance*.  
- **Domain extension**: Covers math, programming, and general tasks; can be mixed with other corpora for broader coverage.  

---

## Limitations

- **Domain coverage**: Strong in mathematics, programming, and general knowledge, but weaker in legal and financial domains.  
- **Output verbosity**: Some reasoning chains are very long, which may lead to verbose answers when training smaller models.  
- **Potential noise**: A small number of samples may still contain teacher hallucinations or redundant reasoning.  

It is recommended to perform additional cleaning or truncation depending on downstream needs.  

---

## License

- **Dataset License**: Apache-2.0  
- **Teacher Model License**: Follow Qwen official terms  

---

## How to Load

```python
from datasets import load_dataset

ds = load_dataset("Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k", split="train")
print(ds[0])
```

---

## Citation

If you use this dataset in your research or product, please cite:  

```bibtex
@dataset{jackrong_zh_qwen3_thinking_distill_100k_2025,
  title   = {Chinese-Qwen3-235B-Thinking-2507-Distill-100k},
  author  = {Soren},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k},
  note    = {Chinese reasoning distillation dataset with Input, CoT_content, and Answer_content fields}
}
```

---

## Acknowledgements & References

This dataset builds upon multiple open-source projects and research efforts. We sincerely thank the following contributors:  

- **Cong Liu et al.** – *Chinese dataset distilled from DeepSeek-R1-671b*  
- **m-a-p/neo_sft_phase2**  
- **hfl/stem_zh_instruction**
- **meta-math/GSM8K_zh**  
- **EduChat-Math**
- **Haijian/Advanced-Math**  
- **gavinluo/applied_math**  

> This project reorganized, cleaned, and distilled these sources, resulting in **Chinese-Qwen3-235B-Thinking-2507-Distill-100k**. We deeply appreciate the contributions of the community and researchers.  

