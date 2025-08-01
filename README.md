<div align="center">
  <h2><b>(ICLR'25 Spotlight) Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/Time-MoE/Time-MoE?color=green)
![](https://img.shields.io/github/stars/Time-MoE/Time-MoE?color=yellow)
![](https://img.shields.io/github/forks/Time-MoE/Time-MoE?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2409.16040">Paper Page</a>]**
**[<a href="https://mp.weixin.qq.com/s/LaYn0IJAOlN9Ufp_qus96Q">中文解读</a>]**

</div>

<p align="center">

<img src="./figures/timemoe-logo.png" width="70">

</p>


> 1️⃣ Time-MoE is the **first work** to scale time series foundation models up to **2.4 billion** parameters, trained from
> scratch.

> 2️⃣ Time-300B is the **largest** open-access time series data collection comprising over **300 billion** time points across >9 domains.

## TODO List
- [ ] Add covariate support
- [ ] Enable fine-tuning of Time-MoE for forecasting with dynamic features
- [x] Support time series classification (sequence and token classification)

## Updates/News:

🚩 **News** (Feb 2025): Time-MoE has been accpeted by ICLR 2025 as a Spotlight (Top 5.1%)!

🚩 **News** (Oct 2024): Time-MoE introduction in [Chinese](https://mp.weixin.qq.com/s/LaYn0IJAOlN9Ufp_qus96Q)

🚩 **News** (Oct 2024): [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) dataset is now available 
on 🤗 Hugging Face

🚩 **News** (Oct 2024): [Time-MoE (base)](https://huggingface.co/Maple728/TimeMoE-50M) and [Time-MoE (large)](https://huggingface.co/Maple728/TimeMoE-200M) are made available
on 🤗 Hugging Face

🚩 **News** (Sept 2024): Time-MoE preprint has been made available on [arXiv](https://arxiv.org/pdf/2409.16040)

## Introduction

Time-MoE comprises a family of decoder-only time series foundation models with a mixture-of-experts architecture,
designed to operate in an auto-regressive manner, enabling universal forecasting with arbitrary prediction horizons and
context lengths of up to 4096.

<p align="center">
    <img src="figures/time_moe_framework.png" alt="" align="center" width="700px" />
</p>

## 📚 Training Data

[Time-300B dataset](https://huggingface.co/datasets/Maple728/Time-300B) is available on 🤗 Hugging Face.

Here's an example of how to use this dataset:
```python
import random
from time_moe.datasets.time_moe_dataset import TimeMoEDataset

ds = TimeMoEDataset('Time-300B')
seq_idx = random.randint(0, len(ds) - 1)
seq = ds[seq_idx]
```

This code snippet shows how to load a random data sequence from the Time-300B dataset. First, download the dataset to the local 'Time-300B' folder, import the TimeMoEDataset class from time_moe.datasets, instantiate the class, and finally retrieve a sequence using a random index.

## 🚀 Getting Started

### Installation

1. Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

**Note: Time-MoE requires `transformers==4.40.1` .**

2. [Optional but **recommended**] Install [flash-attn](https://github.com/Dao-AILab/flash-attention) for faster training and inference speeds with reduced memory usage.

```shell
pip install flash-attn==2.6.3
```

or

```shell
pip install packaging
pip install ninja
# Replace "64" with the number of CPU cores available on your machine for faster compilation
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation
```

### 📈 Making Forecasts

**Note**: The `max_position_embeddings` for Time-MoE is set to during training. This means the maximum sequence length for Time-MoE is **4096**. To achieve optimal forecasting performance, it is recommended that **the sum of `context_length` and `prediction_length` does not exceed 4096.**
If you wish to support longer sequence length, please fine-tune Time-MoE with the desired longer sequence length.

```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

# normalize seqs
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]

# inverse normalize
predictions = normed_predictions * std + mean
```

+ If the sequences are normalized already:

```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
normed_seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]
```

### Evaluation

+ Prepare the benchmark datasets.

You can access the well pre-processed datasets
from [[Google Drive]](https://drive.google.com/drive/folders/1KjnAYr9X3D-jyJpo4yM7Giyq5V1Hga_7?usp=sharing), then place
the downloaded contents under `./dataset`.

+ [Example] Running the follow command to evaluate on ETTh1.

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```

## 🔥 Fine-tuning Time-MoE

### Preparing Your Dataset

To start fine-tuning Time-MoE, your dataset should be converted into a `jsonl` format. Each line represents a time-series data as a dictionary object, where the `sequence` field contains a list of time-series observations. For example:

```jsonl
{"sequence": [1.0, 2.0, 3.0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...]}
```

You have the flexibility to save your converted data in `jsonl`, `json`, or `pickle` format. If you are using the [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) dataset, you can proceed without any additional preprocessing.

### Training Time-MoE on Your Dataset

**Note: If your dataset is small, it is recommended to set `stride` to `1` by adding `--stride 1` to your training command.**

**CPU**

For training with cpu, execute the following command and ensure to replace `<data_path>` with the path to your prepared dataset:

```bash
python main.py -d <data_path>
```

**Single Node with Single or Multiple GPUs**

To leverage a single GPU or multiple GPUs on a single node, use this command:

```bash
python torch_dist_run.py main.py -d <data_path>
```

**Multi-Nodes Multi-GPUs**

For training across multiple nodes, additional environment configurations are necessary to facilitate inter-node communication:

```bash
export MASTER_ADDR=<master_addr>
export MASTER_PORT=<master_port>
export WORLD_SIZE=<world_size>
export RANK=<rank>

python torch_dist_run.py main.py -d <data_path>
```
## 🔥 Fine-tuning Time-MoE for Classification Tasks

Time-MoE supports both **sequence classification** (assigning a single label to the entire time series) and **token classification** (assigning labels to each time step). The data format remains the same as forecasting, with an additional `label` field.

> 📁 **See [`classification_examples/`](classification_examples/) for complete examples, data preparation scripts, and inference utilities.**

**Sequence Classification Data Format**
For sequence classification, each time series gets one label:
```jsonl
{"sequence": [1.0, 2.0, 3.0, ...], "label": 0}
{"sequence": [11.0, 22.0, 33.0, ...], "label": 1}
```

**Token Classification Data Format**
For token classification, each time step gets a label:
```jsonl
{"sequence": [1.0, 2.0, 3.0, ...], "label": [0, 1, 0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...], "label": [1, 0, 1, ...]}
```

**Generate Sample Data**
To create sample classification datasets for testing:
```bash
python classification_examples/create_sample_classification_data.py
```
This will create `sample_data/sequence_classification.jsonl` and `sample_data/token_classification.jsonl` with synthetic data for testing.

**Additional Utilities:**
- `classification_examples/prepare_classification_data.py` - Data preparation utilities and examples
- `classification_examples/inference_classification.py` - Inference script for trained classification models

**Training Commands**

For **sequence classification**:
```bash
python torch_dist_run.py main.py -d <data_path> --task_type sequence_classification --num_classes <num_classes>
```

For **token classification**:
```bash
python torch_dist_run.py main.py -d <data_path> --task_type token_classification --num_classes <num_classes>
```

**Additional Classification Arguments:**
- `--classifier_dropout`: Dropout rate for classification head (default: 0.1)
- `--freeze_backbone`: Freeze the Time-MoE backbone and only train the task-specific head (recommended for small datasets or domain-specific fine-tuning)

**Detailed Training Examples:**

1. **Sequence Classification (e.g., Pattern Recognition)**:
```bash
python main.py -d sample_data/sequence_classification.jsonl \
  --task_type sequence_classification \
  --num_classes 3 \
  --micro_batch_size 4 \
  --train_steps 50 \
  --learning_rate 1e-4 \
  --precision bf16 \
  --attn_implementation eager
```

2. **Token Classification (e.g., Positive/Negative Detection)**:
```bash
python main.py -d sample_data/token_classification.jsonl \
  --task_type token_classification \
  --num_classes 2 \
  --micro_batch_size 4 \
  --train_steps 50 \
  --learning_rate 1e-4 \
  --precision bf16 \
  --attn_implementation eager
```

3. **With Backbone Freezing (for small datasets)**:
```bash
python main.py -d sample_data/sequence_classification.jsonl \
  --task_type sequence_classification \
  --num_classes 3 \
  --freeze_backbone \
  --micro_batch_size 4 \
  --train_steps 30 \
  --learning_rate 1e-3 \
  --precision bf16 \
  --attn_implementation eager
```

4. **Multi-GPU Training**:
```bash
python torch_dist_run.py main.py -d sample_data/sequence_classification.jsonl \
  --task_type sequence_classification \
  --num_classes 3 \
  --global_batch_size 32 \
  --train_steps 100 \
  --precision bf16 \
  --attn_implementation eager
```

**Example with all arguments:**
```bash
python torch_dist_run.py main.py -d <data_path> --task_type sequence_classification --num_classes 5 --classifier_dropout 0.2 --freeze_backbone
```

**Pooling Strategies for Sequence Classification**

Time-MoE supports multiple pooling strategies to aggregate timestep representations into a single sequence representation for classification. Choose the strategy based on your data characteristics:

| Strategy | Description | Best For | Example Use Cases |
|----------|-------------|----------|-------------------|
| `last_token` | Uses the final timestep representation | Sequential patterns, temporal dependencies, state evolution | Stock price trends, speech recognition final states |
| `mean` | Average of all timestep representations | Global statistics, overall signal characteristics, stable patterns | Average heart rate, overall sentiment analysis |
| `max` | Maximum activation across all timesteps | Peak detection, anomaly identification, spike patterns | Seizure detection, network intrusion detection |
| `attention` | Learns which timesteps are most important | Variable-length patterns, complex temporal relationships, multi-scale features | Document classification, irregular heartbeats |
| `multi_scale` | Combines mean, max, and last token | Complex patterns, robust classification, unknown pattern types | General-purpose classification, exploratory analysis |
| `weighted_temporal` | Recent timesteps weighted more heavily | Recent events more important, trend analysis, recency bias | Real-time monitoring, recent trend classification |
| `conv_pool` | 1D convolution followed by pooling | Local patterns, feature extraction, translation invariance | Pattern recognition, motif detection |

**Using Different Pooling Strategies:**
```bash
# Using attention pooling for complex patterns
python main.py -d <data_path> --task_type sequence_classification --pooling_strategy attention

# Using multi_scale for robust performance (recommended default)
python main.py -d <data_path> --task_type sequence_classification --pooling_strategy multi_scale

# Using last_token for sequential dependencies
python main.py -d <data_path> --task_type sequence_classification --pooling_strategy last_token
```

**Performance Tips:**
- 🔍 **Try multiple strategies**: Different datasets may benefit from different approaches
- 📊 **Use multi_scale as baseline**: Combines multiple pooling methods for robust performance  
- ⚡ **Start with simple strategies**: `last_token` and `mean` are fast and often effective
- 🎯 **Consider your domain**: Medical signals vs. financial data may need different strategies
- 📈 **Monitor validation metrics**: Use early stopping to compare strategies fairly
- 🔄 **Ensemble different strategies**: Train multiple models and combine predictions

**Data Format Notes:**
- **Sequence Classification**: Each sequence gets ONE label (integer: 0, 1, 2, ...)
- **Token Classification**: Each timestep gets ONE label (list of integers matching sequence length)
- Use `-100` in token classification labels for positions to ignore in loss calculation
- Supported formats: JSONL, JSON, and pickle files

**Freeze Backbone for Forecasting:**
The `--freeze_backbone` option is also available for forecasting tasks, useful for fine-tuning on domain-specific data:
```bash
python torch_dist_run.py main.py -d <data_path> --task_type forecasting --freeze_backbone --learning_rate 1e-3
```

To train Time-MoE **from scratch**, simply include the `--from_scratch` argument in your command. Here's how it should look:

```bash
python torch_dist_run.py main.py -d <data_path> --from_scratch
```

To explore additional command-line arguments and their usage, invoke the help command:

```bash
python main.py --help
```

## Citation

> 🙋 Please let us know if you find out a mistake or have any suggestions!

> 🌟 If you find the Time-MoE models helpful in your research, please consider to star this repository and cite the
> corresponding [paper](https://arxiv.org/pdf/2409.16040):

```
@misc{shi2024timemoe,
      title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts}, 
      author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
      year={2024},
      eprint={2409.16040},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16040}, 
}
```

## Related Resources
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in arXiv 2024. [\[paper\]](https://arxiv.org/abs/2410.16032) [\[GitHub Repo\]](https://github.com/kwuking/TimeMixer)
* Towards Neural Scaling Laws for Time Series Foundation Models, arXiv 2024. [\[paper\]](https://arxiv.org/pdf/2410.12360)
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML*
  2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *TPAMI*
  2024. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* Transformers in Time Series: A Survey, in *IJCAI*
  2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *TPAMI* 2024. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-LLM [\[repo\]](https://github.com/KimMeen/Time-LLM)
- TimeMixer [\[repo\]](https://github.com/kwuking/TimeMixer)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)
- Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal
  Data [\[repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

## License

This project is licensed under the Apache-2.0 License.
