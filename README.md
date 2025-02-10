
# ClinicInteract

**A Multi-Agent Interactive Benchmark and RAG-Enhanced Diagnostic Agent Framework for Realistic Clinical Simulations**

---

## Overview

**ClinicInteract** is an interactive clinical simulation benchmark designed to evaluate and enhance the diagnostic capabilities of large language models (LLMs) in more realistic healthcare scenarios. It uses multiple specialized agents—including a Patient Agent, Doctor Agent, Laboratory Agent, and Medical Assistant—to simulate the process of clinical consultation, from initial patient interviews to final diagnostic decisions.  

### Key Objectives

1. **Interactive Evaluation**  
   Move beyond static multiple-choice benchmarks by simulating the dynamic, multi-round information-gathering process in real clinical environments.

2. **Realistic Patient Simulation**  
   Incorporate diverse patient characteristics—such as distrust, memory errors, and non-scientific beliefs—to assess an AI’s robustness under non-ideal conditions.

3. **RAG-Enhanced Diagnostic Agent**  
   Use Retrieval-Augmented Generation (RAG) to reduce hallucinations and improve factual consistency during diagnosis.

4. **Modular & Extensible**  
   Provide a framework that can be easily adapted to new datasets, additional patient features, or more sophisticated agent architectures.

---

## Framework Architecture

The ClinicInteract framework consists of:

1. **Patient Agent**  
   - Simulates real patients, holding all patient-specific data (symptoms, medical history, etc.).  
   - Can exhibit various “Patient Characters” (e.g., memory errors, emotional bias) to provide more realistic challenges.

2. **Doctor Agent**  
   - The primary subject of evaluation, responsible for gathering information and making diagnostic decisions.
   - Can query the Patient Agent for symptom details and request specific tests from the Laboratory Agent and Doctor Assistant.
   - Integrates an improved workflow (quick vs. slow thinking) and optional RAG for better accuracy.

3. **Laboratory Agent**  
   - Provides lab test results (blood tests, imaging) upon the Doctor Agent’s request.
   - Can handle essential vs. non-essential tests and simulates missing or normal results when needed.

4. **Doctor Assistant**  
   - Returns basic physical examination results (e.g., blood pressure, heart rate).
   - Functions like the Laboratory Agent but for simpler measurements.

5. **Evaluator**  
   - Automatically compares the Doctor Agent’s predicted diagnosis with the true condition.
   - Offers multiple evaluation metrics, including accuracy and customizable scoring functions.

---

## RAG Integration

A **RAG (Retrieval-Augmented Generation)** system based on the PubMed database and the BM25 retrieval algorithm was designed to enhance the agent's accuracy in medical diagnosis.
- **BM25** (Best Matching 25) is a classic information retrieval algorithm, an enhanced version of the Okapi TF-IDF algorithm.
- **PubMed**, maintained by the U.S. National Library of Medicine, is one of the most authoritative biomedical literature databases in the world.
- useage: 
- Open main.py, modify use_RAG=True in:
    ```python
    doctor = Doctor(max_conversation=total_inferences, backend=self.doctor_backend, use_RAG=True)
    ``` 


## Getting Started

### Prerequisites
- Python 3.8+  (I use Python 3.8.19)
- Your preferred LLM API (OpenAI, etc.) 
- Recommended API (This is the most cost-effective API I have tested): 
Silicon Flow Deepseek V3/R1  
Registration link: https://siliconflow.cn/
- The best way to use the Tongyi Qianwen series models is through Alibaba Cloud's Bailian API. You can register at the following address:  
https://cn.aliyun.com/product/bailian?from_alibabacloud=
- In src/ClinicInteract/config.py, put all API keys in the correct places.
#Add a more minimal Conda environment later.

### Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/JacobChangUCL/Final_Project.git
    ```
2. **Install the required packages**s
    ```bash
    conda env create -f environment.yml
    ```
3. **Activate the environment**
    ```bash
    conda activate ClinicInteract
    ```
4. **Open the Final_Project in your IDE**  


5. **Click on `src/ClinicInteract/main.py` ,or click `src/ClinicInteract/baseline.py`**
to run baseline.
### Project Structure
```plaintext
Final_Project/
├── data/
├── notebooks/
├── src/
│   ├── ClinicInteract/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── config.py
│   │   ├── doctor.py
│   │   ├── doctor_information_pressure.py    #an unfinished prototype and is not included in this project.
│   │   ├── Doctor_RAG.py
│   │   ├── laboratory.py
│   │   ├── main.py
│   │   ├── patient.py
│   │   ├── physical_examination_findings.py
│   │   ├── prompt_dict.py
│   │   ├── prompts.txt      #chinese version prompts.Deprecated.
│   │   ├── Rag_optimizer.py
│   │   ├── test.py
│       ├── utils.py
├── ──── datasets/
│           filtered_medqa_test_set.jsonl
│           filtered_medqa_test_set_final_version.jsonl
│           filtered_medqa_test_set_updated.jsonl
├── RAG/
│   ├── corpus/
│   ├── data/
│   ├── __init__.py
│   ├── rag.py
│   ├── template.py
│   ├── utils_RAG.py
```