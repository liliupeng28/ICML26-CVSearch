# CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception

<p align="center">
<a href="http://arxiv.org/abs/2605.23655"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper-Arxiv"></a>
<a href="https://icml.cc/virtual/2026/poster/65958"><img src="https://img.shields.io/badge/Conference-ICML%202026-green" alt="Conference-ICML 2026"></a>
</p>

Official PyTorch implementation of the **ICML 2026** paper: "**CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception**". 

**CVSearch** is a training-free, adaptive framework designed to solve the trade-off between coverage and efficiency in high-resolution (HR) image perception. By mimicking human cognitive search patterns, CVSearch dynamically schedules search strategies via an *Assess-then-Search* workflow, leveraging both visual experts and an innovative semantic-aware scanning mechanism.

![CVSearch Overview](figures/CVSearch.png) 
*Overview of the CVSearch framework: Integrating Expert-Assisted Search with Semantic Guided Adaptive Patching and Dynamic Bottom-Up Search.*

## 🚀 News
- **[2026-05]** 🎉 **CVSearch** has been accepted by **ICML 2026**!
- **[2026-05]** 🔥 Full evaluation codebase, model wrappers for LLaVA-OneVision, Qwen2.5/3-VL, and InternVL2.5 are released.

## 💡 Key Highlights
- **Assess-then-Search Workflow**: Intelligently balances efficiency and accuracy by invoking expert-assisted search for global context and triggering scanning only upon failure.
- **Semantic Guided Adaptive Patching (SGAP)**: Moves beyond rigid grid partitioning to decompose images into semantically consistent regions, effectively mitigating object fragmentation.
- **Dynamic Bottom-Up Search**: Driven by a **Visual Complexity** prior, this strategy enables precise iterative exploration of local details while strictly controlling computational redundancy via branch pruning.
- **Training-Free & Scalable**: A plug-and-play solution that enhances models from 2B up to 32B parameters (e.g., Qwen3-VL-32B) without any fine-tuning.

---

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/liliupeng28/ICML26-CVSearch.git
cd ICML26-CVSearch
```
**2. Create environment and install dependencies:**
```bash
conda create -n cvsearch python=3.10 -y
conda activate cvsearch
pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Data & Model Preparation

**1. Evaluation data**

The core evaluation data (including V* Bench, HR-Bench, MME-RealWorld-Lite, TreeBench, FineRS-
4K) is provided [here](https://www.modelscope.cn/datasets/llp1995/hr_data). After downloading, please unzip it and its path is referred as to datasets/hr_data.

**2. Model checkpoints**

In our work, we implement CVSearch with LLaVA-OneVision(ov), InternVL2.5, and Qwen2.5/3-VL series, you could download these checkpoints before running or automatically download them when executing the from_pretrained method in transformers. In addition, you need to download checkpoints of [SAM 3](https://www.modelscope.cn/models/llp1995/sam3) and [spacy](https://www.modelscope.cn/models/llp1995/sam3) Place the checkpoint in the models/ folder.
* [LLaVA-ov-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)
* [InternVL2.5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)
* [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
* [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
* [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
* [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
* [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
* [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

## 📊 Quick Start (Evaluation)
Use our customizable script run_eval_cvsearch.sh to reproduce the paper results.

**1. Run Standard Evaluation**
Modify run_eval_cvsearch.sh to Direct Answer mode and set ROOT_PATH.
```bash
cd CVSearch/cvsearch
bash run_eval_cvsearch.sh --model_path models/Qwen3-VL-8B --benchmark vstar --gpu_id 0

# Get the result
python eval/eval_results_vstar.py --answers-file ROOT_PATH/CVSearch/cvsearch/CVsearch/eval/answers/vstar/Qwen3-VL-2B/direct_answer.jsonl
```

**2. Custom Model & Benchmark**
Modify ROOT_PATH in run_eval_cvsearch.sh to your working directory and set ROOT_PATH.
```bash
cd CVSearch/cvsearch
bash run_eval_cvsearch.sh --model_path models/Qwen3-VL-8B --benchmark vstar --gpu_id 0

# Get the result
python eval/eval_results_vstar.py --answers-file ROOT_PATH/CVSearch/cvsearch/CVsearch/eval/answers/vstar/Qwen3-VL-2B/cvsearch.jsonl
```
Supported benchmarks: vstar, hr-bench_4k, hr-bench_8k, mme-realworld-lite, treebench, fines-bench.

## 📈 Main Results

**1. Scaling with Different MLLM Backbone Sizes**

| Method | V* | HR-4K | HR-8K |
| :--- | :---: | :---: | :---: |
| Qwen2.5-VL-3B | 77.0 | 65.9 | 62.9 |
| **- w/ CVSearch** | **91.1** | **70.5** | **67.3** |
| Qwen3-VL-2B | 79.1 | 71.3 | 67.8 |
| **- w/ CVSearch** | **92.2** | **74.0** | **73.6** |
| Qwen3-VL-4B | 89.5 | 76.3 | 71.6 |
| **- w/ CVSearch** | **93.7** | **77.4** | **75.1** |
| Qwen3-VL-8B | 88.0 | 79.1 | 74.5 |
| **- w/ CVSearch** | **91.6** | **79.5** | **76.5** |
| Qwen3-VL-32B | 86.9 | 76.5 | 70.9 |
| **- w/ CVSearch** | **89.5** | **80.3** | **78.4** |

**2. Search Efficiency Comparison**

| Visual Search Method | V* Bench (Acc) | V* Bench (Thr) | HR-Bench 4K (Acc) | HR-Bench 4K (Thr) | HR-Bench 8K (Acc) | HR-Bench 8K (Thr) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B | 71.2 | 8.30 | 68.8 | 7.62 | 65.3 | 7.62 |
| - w/ Visual Expert (SAM 3) | 84.3 | 3.60 | 71.9 | 5.59 | 68.1 | 5.30 |
| - w/ Scan-based (Zoom Eye) | 85.3 | 0.68 | 72.5 | 1.29 | 69.8 | 0.68 |
| - w/ Scan-based (RAP) | 84.8 | 0.66 | 74.8 | 1.22 | 76.0 | 0.58 |
| **- w/ CVSearch (Ours)** | **90.1** | **1.02** | **76.6** | **3.77** | **75.6** | **1.92** |

## 📝 Acknowledgments

Our implementation is built upon the foundational architectures of [ZoomEye](https://github.com/om-ai-lab/ZoomEye), and [SAM 3](https://github.com/facebookresearch/sam3). We sincerely thank the authors for open-sourcing their incredible works.

## Contact

If you have any question, you can raise an issue or email Liupeng Li (25b951045@stu.hit.edu.cn).
