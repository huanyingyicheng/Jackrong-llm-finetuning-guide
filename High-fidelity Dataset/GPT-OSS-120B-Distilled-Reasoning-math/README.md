---
license: apache-2.0
task_categories:
- question-answering
- summarization
- table-question-answering
language:
- en
pretty_name: Jackrong/GPT-OSS-120B-Distilled-Reasoning-math
size_categories:
- 1K<n<10K
---
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/Z_e-AT-WC-W1FyrvBhPS1.jpeg)
# GPT-oss-120B-Distilled-Reasoning-math Dataset
**Data Source Model**: **gpt-oss-120b**  
**Task Type**: Mathematical Problem Solving  
**Data Format**: JSON Lines
**Fields**: Generator, Category, Input, CoT_Native_Reasoning, Reasoning, Answer  

---

#  Core Statistics
Generated complete reasoning processes and answers using **gpt-oss-120b** (MXFP4).  
The text length of the dataset reflects the depth and complexity of its content. I have statistically analyzed the lengths of the **input** (question), **Reasoning**, and **Answer**.  
To understand the data distribution more intuitively, I performed some visualization analysis.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/vhb8BN37WVllGjHrmFR1i.png)

---

##  Quality and Content Evaluation
This evaluation did not introduce an LLM scoring model. Instead, two custom quantitative metrics were used to assess data structure and reasoning characteristics:

- **Reasoning Complexity Ratio**: **39.19**  
  *Calculation Method*: Average reasoning characters ÷ Average input characters  
  *Meaning*: Measures the extent of the model's reasoning chain. A higher value means the model provides sufficient reasoning details even for short questions.

- **Answer Efficiency Ratio**: **0.67**  
  *Calculation Method*: Average answer words ÷ Average reasoning words  
  *Meaning*: Measures the refinement from reasoning to the answer. A lower value indicates that the reasoning is divergent, while the answer is convergent and concise.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/GNkBaNsuasJ4Jal9w0ABm.png)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/3tQm0sHGSxeTU1ZverdQk.png)  

---

##  Comprehensive Evaluation
The dataset demonstrates high-quality mathematical problem-solving capabilities, featuring:
- **Comprehensive Reasoning Chain**: Detailed thought processes and clear logical steps.
- **Rich Mathematical Expression**: Effective use of LaTeX for formula typesetting.
- **Balanced Input-Output Relationship**: The complexity of the reasoning process is reasonably correlated with the complexity of the problem.

---

##  Dataset Structure
**File Format**: .jsonl (one sample per line, independent JSON object)
To make training easier for everyone, I have prepared various data structure templates, offering three common annotation types for different distillation and cleaning logic.

**1. Standard JSON Structure**
To facilitate the training of reasoning models or the creation of SFT data, explicitly separate the chain of thought and the final answer in the output. 
```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "Given that 2^x = 8, find x.",
  "CoT_Native_Reasoning": "We note that 8 = 2^3...",
  "answer": "The answer is 3."
}
```

**2. OpenAI Harmony**
Messages enclosed by tags <start>|user|message|...<end> and <start>|assistant|...<end>, aligning with the OpenAI Harmony style.
```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "<start>|user|message|>In triangle ABC with BC=3, ... <end>",
  "output": "<start>|assistant|We have a right triangle at C, ... <end>"

}
```
**3. Think**
The format is like the Qwen3 series model and DeepSeek.

```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "Solve: If 12x = 36, what is x?",
  "output": "[think]First, divide both sides by 12. 36 / 12 = 3. So x = 3.[/think] The answer is 3."
}
```
---

## Training and Usage Recommendations
- **Alignment Training**: For CoT training, please ensure the template labels are suitable for the model.
- **Evaluation**: Report reasoning accuracy with/without CoT simultaneously; provide an "answer-in-the-box" parser to stabilize numerical extraction.
- **Safety Thresholds**: Prioritize quality over quantity for erroneous/inconsistent samples; set safety upper bounds for long samples and process them in chunks.

---

##  Acknowledgements
The construction of this dataset is based on the generation capabilities of **gpt-oss-120b** and the optimized design of mathematical reasoning templates.  
Special thanks to the open-source community for their contributions in **mathematical expression formatting**, **data cleaning scripts**, and **visualization analysis**.  

**Seed Questions**: Derived in part from *nvidia/Nemotron-Post-Training-Dataset-v1*.  
**License**: CC-BY-4.0  

**Dataset Citation**:
```
@dataset{jackrong_2025_gpt_oss_math_distill,
  title   = {GPT-OSS-120B-Distilled-Reasoning-math},
  author  = {Jackrong},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/GPT-OSS-120B-Distilled-Reasoning-math}
}
```

---

# 📚 数据集概览
数据源模型: **gpt-oss-120b**  
任务类型: **Mathematical Problem Solving**  
数据格式: **JSON Lines (.jsonl)**  
字段: **Generator, Category, Input, CoT_Native_Reasoning, Reasoning, Answer**  

---

## 📈 核心统计指标
使用 **gpt-oss-120b**（MXFP4 格式）生成完整的推理过程与答案。  
数据集的文本长度反映了其内容的深度和复杂性。我对输入（问题）、Reasoning 和 Answer 的长度进行了详细统计。  
为了更直观地理解数据分布，我进行了可视化分析。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/vhb8BN37WVllGjHrmFR1i.png)

---

## ⭐ 质量与内容评估
本次没有引入 LLM 评分模型，而是使用两项自定义量化指标评估数据结构与推理特性：

- **推理复杂度比率**（Reasoning Complexity Ratio）：39.19  
  *计算方式*：平均推理字符数 ÷ 平均输入字符数  
  *含义*：衡量模型推理链的展开程度。较高值意味着即使面对简短题目，模型也能提供充分的推理细节。

- **答案效率比率**（Answer Efficiency Ratio）：0.67  
  *计算方式*：平均答案词数 ÷ 平均推理词数  
  *含义*：衡量推理到答案的精炼程度。较低值代表推理是发散的，而答案是收敛简洁的。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/GNkBaNsuasJ4Jal9w0ABm.png)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/3tQm0sHGSxeTU1ZverdQk.png)  

---

## ✅ 综合评估
数据集展现了高质量的数学问题解决能力，具有：
- **全面的推理链**：思维过程详尽，逻辑步骤清晰。
- **丰富的数学表达**：能够有效利用 LaTeX 进行公式排版。
- **均衡的输入输出关系**：推理过程的复杂性与问题的复杂性合理相关。

---

## 🏗️ 数据集结构
**文件格式**：.jsonl（每行一个样本，独立 JSON 对象）

**示例**：
**1. Standard JSON Structure**
To facilitate the training of reasoning models or the creation of SFT data, explicitly separate the chain of thought and the final answer in the output. 
```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "Given that 2^x = 8, find x.",
  "CoT_Native_Reasoning": "We note that 8 = 2^3...",
  "answer": "The answer is 3."
}
```

**2. OpenAI Harmony**
Messages enclosed by tags <start>|user|message|...<end> and <start>|assistant|...<end>, aligning with the OpenAI Harmony style.
```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "<start>|user|message|>In triangle ABC with BC=3, ... <end>",
  "output": "<start>|assistant|>We have a right triangle at C, ... <end>"

}
```
**3.Think**
Using ｜think｜and ｜/think｜ package reasoning content and add answer behind it directly.
The format is like The Qwen3 series model and DeepSeek.
```json
{
  "generator": "gpt-oss-120b",
  "category": "math",
  "Input": "Solve: If 12x = 36, what is x?",
  "output": "<think>First, divide both sides by 12. 36 / 12 = 3. So x = 3.</think> The answer is 3."
}
```

---

## 📌 训练与使用建议
- **对齐训练**：CoT 训练请确保模板标签适合模型。
- **评测**：同时报告含/不含 CoT 的推理正确率；提供“盒中答案”解析器以稳定提取数值。
- **安全阈**：错误/不一致样本宁缺毋滥；对长样本设置安全上限并分块处理。

---

## 🙏 致谢
本数据集的构建基于 **gpt-oss-120b** 的生成能力以及数学推理模板的优化设计。  
特别感谢开源社区在 **数学公式排版**、**数据清洗脚本** 和 **可视化分析** 方面的贡献与支持。  

**种子问题来源**：部分来自 *nvidia/Nemotron-Post-Training-Dataset-v1*。  
**许可协议**：CC-BY-4.0  

**数据集引用**：
```
@dataset{jackrong_2025_gpt_oss_math_distill,
  title   = {GPT-OSS-120B-Distilled-Reasoning-math},
  author  = {Jackrong},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/GPT-OSS-120B-Distilled-Reasoning-math}
}
