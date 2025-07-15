<p align="center">
<h1 align="center"> Divide-Solve-Combine: An Interpretable and Accurate Prompting Framework for Zero-shot Multi-Intent Detection</h1>
</p>


<p align="center">
  	<b>
    [<a href="https://ojs.aaai.org/index.php/AAAI/article/view/34688">Paper</a>] | [<a href="https://github.com/LightChen233/DSCP">Code</a>] 
    </b>
    <br />
</p>

🌟 Any contributions via PRs, issues, emails or other methods are greatly appreciated.

## 🔥News
- 🎖️ **Our work is accepted by AAAI 2025.**

## 💡 Motivation
Zero-shot multi-intent detection is capable of capturing multiple intents within a single utterance without any training data, which gains increasing attention.
Building on the success of large language models (LLM), dominant approaches in the literature explore prompting techniques to enable zero-shot multi-intent detection.
While significant advancements have been witnessed, the existing prompting approaches still face two major issues: 
**lacking explicit reasoning** and **lacking interpretability**.
Therefore, in this paper, we introduce a Divide-Solve-Combine Prompting (DSCP) to address the above issues.
Specifically, DSCP explicitly decomposes multi-intent detection into three components including 
(1) **single-intent division prompting** is utilized to decompose an input query into distinct sub-sentences, each containing a single intent;
(2) **intent-by-intent solution prompting** is applied to solve each sub-sentence recurrently; and 
(3) **multi-intent combination prompting** is employed for combining each sub-sentence result to obtain the final multi-intent result.
By decomposition, DSCP allows the model to track the explicit reasoning process and improve the interpretability.
In addition, we propose an interactive divide-solve-combine prompting (Inter-DSCP) to naturally capture the
interaction capabilities of large language models.
Experimental results on two standard multi-intent benchmarks (i.e., MixATIS and MixSNIPS) reveal that both DSCP and Inter-DSCP obtain substantial improvements over baselines, achieving superior performance and higher interpretability.



## 🎯 Installation

### 1. Install from git
DSCP requires `Python>=3.10`.
```bash 
git clone https://github.com/LightChen233/DSCP.git && cd DSCP/
pip install -r requirements.txt
```
### 2. Evaluation for reproduction
```bash
python evaluate.py --dataset_type mix-snips \
                   --prompt_type dscp \
                   --model_name gpt35
```
where the detailed parameter descriptions are as follows:
- `--dataset_type` can be selected from `[mix-atis, mix-snips]`
- `--prompt_type` can be selected from `[dscp, inter-dscp]`
- `--model_name` can be selected from `[gpt35, gpt4, palm2]`

### 3. Evaluation for your results
```bash
python request.py --dataset_type mix-snips \
                   --prompt_type dscp \
                   --model_name gpt-3.5-turbo \
                   --api_key sk-xxx \
                   --request_proxy None
```
where the detailed parameter descriptions are as follows:
- `--dataset_type` can be selected from `[mix-atis, mix-snips]`
- `--prompt_type` can be selected from `[dscp, inter-dscp]`
- `--model_name` can be selected from OpenAI's API verion
- `--api_key` is OpenAI's API key
- `--request_proxy` is the base_url of OpenAI's API
After completing the request, execute the following command
```bash
python evaluate.py --load_path [OUTPUT_PATH]
```

## 🖨️File Structure

```yaml
root
├── data           # data folder where the dataset is loaded
│   ├── MixATIS           # MixATIS test data
│   └── MixSNIPS          # MixSNIPS test data
├── experiment     # All experimental data
│   ├── gpt4              # Output results of GPT-4
│   ├── gpt35             # Output results of GPT-3.5-Turbo
│   └── palm2             # Output results of PaLM-2
├── utils          # Tool library folder
│   ├── common_tool.py    # Some common utility functions
│   ├── metric.py         # Indicator calculation tool
│   └── request_tool.py   # API request tool
├── request.py     # Request script
└── evaluate.py    # Evaluation script
```

## ✒️ Reference
If you find this project useful for your research, please consider citing the following paper:

```
@inproceedings{qin2024dscp,
    title = "Divide-Solve-Combine: An Interpretable and Accurate Prompting Framework for Zero-shot Multi-Intent Detection",
    author = "Qin, Libo  and
      Chen, Qiguang  and
      Zhang, Jin  and
      Fei, Hao  and
      Che, Wanxiang  and
      Li, Min",
    booktitle = "Proc. of AAAI",
    year = "2025",
}
```

## 📲 Contact

Please create Github issues here or email [Libo Qin](mailto:lbqin@csu.edu.cn) or [Qiguang Chen](mailto:charleschen2333@gmail.com) if you have any questions or suggestions. 

