# Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information (PFCDR)

## Introduction
This repository provides the implementations of PFCDR and two popular baselines (TGTOnly, EMCDR):
* TGTOnly：Train a MF model with the data of the target domain.
* EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/Proceedings/2017/0343.pdf) (IJCAI 2017)



## Requirements

- Python 3.6
- Pytorch > 1.0
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json             # Common Configurations
│   ├── inversion_config.json   # Configurations for Inversion stage
│   ├── main.py                 # Entry function
│   ├── models.py               # Models based on MF, GMF or Youtube DNN
│   ├── preprocessing.py        # Parsing and Segmentation
│   ├── inversion.py            # Module for model inversion
│   ├── float_embedding.py      # A new version of torch.nn.embedding that support float as input
│   ├── README.md
│   └── run.py                  # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python main.py --process_data_mid 1 --process_data_ready 1
```

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models within `MF, GMF or DNN`, default for `MF`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default for `[0.8, 0.2]`
- epoch: pre-training and CDR mapping training epoches, default for `10`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.01`
- model_name: base model for embedding, default for `MF`

You can run this model through:

```powershell
# Run directly with default parameters 
python main.py --process_data_mid 0 --process_data_ready 0

```
It will run a little while since it need to generate prototypes on overlapped users in source and target domains. This 
prototypes will automatically save, next time you can directly use some users for CDR.

If you wanna try different `weight decay`, `embedding dimmension` or more tasks, you may change 
the settings in `./config.json`. If you wanna try the parameter of inversion stage you may change 
the settings in `./inversion_config.json`.

## Book to Movie: 20% test users

This run uses the original Python source files at upstream revision
`c88e6fe`: task 1, MF, ratio `[0.8, 0.2]`, seed 2020, 10 epochs,
embedding dimension 10, learning rate 0.01, prototype learning rate 0.005.
The 20% split is over common users, with all their target-domain ratings
held out. It does not mean holding out 20% of individual ratings.

On an RTX 4060 Laptop GPU with 8 GB VRAM, the original inversion batch of
200 passed a forward/backward/Adam memory check (about 4581 MiB peak).
The original evaluation batch of 1024 ran out of GPU memory. Only task 1's
`batchsize_test` is reduced to 128 (about 1487 MiB in the evaluation check,
including a 400 MiB reserve). Training batches, inversion settings, losses,
mapping functions and the scoring/voting algorithm are unchanged.
The MF implementation's existing hard-coded Top-K remains 70000.

Preprocessing uses Python 3.10.20, NumPy 1.26.4, pandas 2.2.3 and tqdm
4.67.1. Training uses Python 3.11.14, PyTorch 2.3.1 with CUDA 11.8,
NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.2 and tqdm 4.67.1.
Python 3.10 is used for preprocessing to retain the original set sampling
without editing the source. Set `$prepPython` and `$trainPython` to the
respective environment executables before running these PowerShell commands
from the repository directory:

```powershell
$env:PYTHONHASHSEED = '2020'
$env:CUDA_VISIBLE_DEVICES = '0'
& $prepPython -c "import json; from preprocessing import DataPreprocessingMid; c=json.load(open('config.json')); DataPreprocessingMid(c['root'],'Books').main()"
& $prepPython -c "import json; from preprocessing import DataPreprocessingMid; c=json.load(open('config.json')); DataPreprocessingMid(c['root'],'Movies_and_TV').main()"
& $prepPython -c "import json,random; from preprocessing import DataPreprocessingReady; random.seed(2020); c=json.load(open('config.json')); DataPreprocessingReady(c['root'],c['src_tgt_pairs'],'1',[0.8,0.2]).main()"
git rev-parse HEAD
& $trainPython -u main.py --task 1 --base_model MF --gpu 0 --seed 2020 --epoch 10 --lr 0.01 --lr_prototype 0.005
```

The data-preparation commands call the original classes for this task only.
The training command retains the original `main.py -> Run.main()` flow.
Omit the `process_data_*` flags during training: the original parser treats
an explicitly passed string `0` as truthy. The ratio is the original default.

Record the Git commit, full commands, environment versions and console log
for each run. A single run yields the source code's per-epoch minimum test
MAE and RMSE; these two minima may occur in different epochs. It is not the
paper's ten-run mean. For independent repetitions, use different training
seeds and separate model/prototype cache directories, and commit each
parameter change before running. The paper does not publish the exact
split/checkpoint-selection/seeds needed to guarantee the same table values.

### Recorded single run (seed 2020)

The complete run is recorded in
[`results/book_movie_20_seed2020.json`](results/book_movie_20_seed2020.json),
including every epoch, dataset checksums, settings and validation results.

| Method | Minimum MAE | MAE epoch | Minimum RMSE | RMSE epoch |
| --- | ---: | ---: | ---: | ---: |
| TGTOnly | 4.150802 | 10 | 4.740552 | 10 |
| EMCDR | 1.125340 | 8 | 1.425091 | 8 |
| PFCDR (upstream `sfcdr` keys) | 0.943425 | 1 | 1.226853 | 2 |

These are independently selected minima from one run, following the original
`update_results()`. The paper's PFCDR values, 0.9579 / 1.2437, are a ten-run
mean. The locally saved last-epoch PFCDR checkpoint has MAE 0.956729 and RMSE
1.227226; it is distinct from the independently selected best metrics above.

The split contains 29,910 training common users, 7,478 test common users and
170,081 test ratings. All three prototype files have the expected unique
users and ten finite prototype coordinates. Comparing the baseline and
last-epoch checkpoints confirms that source embeddings, target embeddings
and the EMCDR mapping stayed unchanged during PFCDR; only `rp_mapping`
was updated.

The first process encountered a CSV sharing `PermissionError` during target
prototype output. The completed source prototypes and baseline checkpoint
were retained. Target prototypes were rebuilt with the original inversion
class after replaying the original random sequence; all 1,200 previously
written target rows matched exactly. Progress reads were changed to allow
concurrent writes. The repository's Python source files remain identical to
`c88e6fe`, and the experiment settings did not change during recovery.

## MF reproduction: three tasks, three test ratios, two seeds

The execution plan is [experiments/full_reproduction_plan.json](experiments/full_reproduction_plan.json).
It covers TGTOnly, EMCDR and PFCDR for Book → Movie, Book → Music and
Music → Movie at 20%, 50% and 80% held-out common users. Each condition uses
two distinct training seeds, 2020 and 2021, on a fixed split created
with seed 2020 and PYTHONHASHSEED=2020. The verified single run above is
reused once as task 1 / 20% / seed 2020, leaving 17 new runs out of 18 total.
The user reduced the plan from ten to two repetitions per condition on
2026-11-16. The active seed-2021 training continues with its existing parameters.

Each run has a separate working directory, baseline checkpoint and prototype
cache. These completed runs called the original Run.main(), with the original
model, optimizers, losses, inversion and voting functions. Auxiliary execution,
monitoring and recovery scripts were outside this repository. At execution time,
all algorithm Python files were identical to c88e6fe. The optional changes below
were added after the 18 runs completed and were not used in those results.

Only evaluation batch sizes are reduced: task 1 remains 128; task 2 changes
from 256 to 128, and task 3 changes from 512 to 128. Original training
batches, inversion batch 200, inversion iterations 300, epoch count 10,
learning rates, embedding dimension and hard-coded MF Top-K are retained.

Before each run, its parameters are committed in
[experiments/current_run.json](experiments/current_run.json). After completion,
the actual per-epoch metrics are saved under results/full_reproduction/trials/
and committed. Completed seeds and statistics are tracked in
[results/full_reproduction/summary.json](results/full_reproduction/summary.json).
Within a run, the original independently selected minimum MAE and RMSE are
used. Across the two completed seeds, their arithmetic mean and sample standard
deviation are calculated. A row is filled only after both seeds finish.
These are two-run reproduction statistics; the paper reports ten-run means.
Failed attempts and repeated launches of the same seed are never counted
as additional independent runs.

<!-- FULL_REPRODUCTION_TABLE_START -->
| Task | Test users | Runs | TGT MAE | TGT RMSE | EMCDR MAE | EMCDR RMSE | PFCDR MAE | PFCDR RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Book → Movie | 20% | 2/2 | 4.1960 | 4.7702 | 1.1127 | 1.4044 | 0.9347 | 1.2129 |
| Book → Movie | 50% | 2/2 | 4.2233 | 4.7825 | 1.1735 | 1.4846 | 0.9821 | 1.2871 |
| Book → Movie | 80% | 2/2 | 4.2363 | 4.8169 | 1.3265 | 1.6790 | 1.1217 | 1.4745 |
| Book → Music | 20% | 2/2 | 4.4849 | 5.1491 | 1.7406 | 2.0606 | 1.7211 | 1.9910 |
| Book → Music | 50% | 2/2 | 4.5409 | 5.2074 | 1.9700 | 2.3023 | 2.0612 | 2.3301 |
| Book → Music | 80% | 2/2 | 4.5394 | 5.2247 | 2.4422 | 2.7858 | 2.5904 | 2.8722 |
| Music → Movie | 20% | 2/2 | 4.0931 | 4.6433 | 1.1192 | 1.4386 | 0.9827 | 1.2717 |
| Music → Movie | 50% | 2/2 | 4.1556 | 4.7193 | 1.1734 | 1.5059 | 1.0249 | 1.3381 |
| Music → Movie | 80% | 2/2 | 4.1675 | 4.7535 | 1.2269 | 1.5919 | 1.0681 | 1.4379 |
<!-- FULL_REPRODUCTION_TABLE_END -->

## 图书到音乐：K 扫描准备（参数核对，尚未启动）

计划见 [book_music_k_sweep_plan.json](experiments/book_music_k_sweep_plan.json)。
仅针对 MF 的图书到音乐任务，拟比较 K=10000、15000、20000、25000、30000，
沿用 20%、50%、80% 测试用户比例，只使用源码默认训练种子 2020。
用户已确认每个比例只独立运行一次，每个训练阶段保留 10 个 epoch，
不启用梯度累积，也不对独立运行或 epoch 取平均。
同一组训练参数分别评估所有 K，避免每个 K 都重复预训练和原型反演。
论文第 5.4.1 节指出该任务在 K=10000～30000 区间效果最好。

新增两个可选接口，原有调用的默认行为保持不变：

- `MFBasedModel.forward(..., topk=20000)` 显式选择该次评估的 K。
  未传此参数时仍使用历史值 70000，不会自动改用配置中的 `self.topk`。
- `Run.train(..., accumulation_steps=2)` 在两个小批次后执行一次 Adam 更新。
  也可通过 `config['accumulation_steps']` 按训练 stage 指定步数。
  默认步数为 1。累积实现按实际样本数加权，最后不足两个批次时也会更新，
  不会把最后一个小批次错误地固定除以 2。

MF 不含 BatchNorm 和 Dropout。相同样本顺序、相同损失和优化器状态下，
512 条样本累积两次的梯度对应 1024 条样本一次更新，允许浮点运算误差。
这不表示与原先每 512 条更新一次的训练轨迹相同，也不能据此声称论文使用了 1024。
公开配置中，该任务的源域、目标域、原型映射训练批次为 512，EMCDR 映射批次为 128。
1024 出现在任务 3 的原型映射配置中。

### 已完成的验证

`tests/test_batch_accumulation_and_k.py` 的 7 项检查通过，涵盖 Adam 更新和状态、
512×2 与 1024、尾部小批次、向量映射损失、默认更新路径，以及显式 K 投票。
这些检查使用合成数据，没有训练 Amazon 实验模型。

显存记录见 [feasibility.json](results/book_music_k_sweep/feasibility.json)。
测试使用已有任务 2、20%、seed=2020 的基础模型和原型，仅读取并前向计算，
没有 backward 或 optimizer.step，也不将本次探测的输出当作推荐准确率。

| K | 评估 batch size | 峰值已分配显存（MiB） |
| ---: | ---: | ---: |
| 10000 | 512 | 1965.35 |
| 15000 | 512 | 1965.35 |
| 20000 | 512 | 1965.35 |
| 25000 | 512 | 1965.35 |
| 30000 | 512 | 1990.19 |

512 和分成四个 128 批次的预测输出在本次探测中相同。
上述显存不包含正式训练时可能同时驻留的数据和优化器状态。
源域、目标域及原型映射的 512 训练批次已在此前任务 2 实验中运行成功。
反演维持 batch=200；反演优化的是每个用户独立的物品权重，
不能把训练阶段的 512×2 累积设置直接套到反演上。

本次按用户最终选择保留源码训练批次：源域 512、目标域 512、
EMCDR 映射 128、原型映射 512，每个批次更新一次参数。
按用户最终要求，除 K 外均按该任务源码参数，测试批次恢复为 256，
不使用先前提议的 512。K 使用上述五个值。
每个 K 分别记录 10 个 epoch 内的最低 MAE、最低 RMSE，二者可能来自不同 epoch。
共 3 组训练配置、15 组 K 评估配置；结果不是两次或十次运行的均值。
2026-11-17，用户查看参数后已批准启动全部三个比例，并明确取消 K=70000 复核。
执行入口为 `experiments/run_book_music_k_sweep.py run`。
它校验并复用对应比例、seed=2020 的源码预训练模型与反演原型，
重新训练原型映射 10 个 epoch。初始化与前序 DataLoader 的 PyTorch 随机状态会重放。
同一批数据的五个 K 共用源码中的一次用户排序，再分别平均前 K 个用户的评分；
该计算已与逐个调用原 MF 前向计算进行一致性检查，未改变匹配或评分规则。
结果按比例写入 `results/book_music_k_sweep/trials/` 并提交推送，
`results/book_music_k_sweep/summary.json` 汇总 15 组实际结果。

梯度累积参考：[PyTorch 2.3 文档](https://docs.pytorch.org/docs/2.3/notes/amp_examples.html#gradient-accumulation)。

If the code is helpful for you, please cite this paper:

```
@inproceedings{wang2025privacy,
  title={Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information},
  author={Wang, Cheng and Xu, Wenchao and Wang, Haozhao and Liu, Wei and Li, Ruixuan},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={450--461},
  year={2025}
}
```


### 图书到音乐 K 扫描实际结果（seed=2020，单次运行）

<!-- BOOK_MUSIC_K_SWEEP_START -->
| 测试比例 | K | seed | MAE | RMSE | MAE 最优 epoch | RMSE 最优 epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 10000 | 2020 | 1.1241 | 1.5398 | 3 | 3 |
| 20% | 15000 | 2020 | 1.1324 | 1.5411 | 3 | 3 |
| 20% | 20000 | 2020 | 1.1509 | 1.5506 | 3 | 3 |
| 20% | 25000 | 2020 | 1.1767 | 1.5653 | 3 | 3 |
| 20% | 30000 | 2020 | 1.2093 | 1.5855 | 3 | 3 |
| 50% | 10000 | 2020 | 1.2535 | 1.7336 | 6 | 7 |
| 50% | 15000 | 2020 | 1.2777 | 1.7454 | 6 | 6 |
| 50% | 20000 | 2020 | 1.3124 | 1.7643 | 6 | 6 |
| 50% | 25000 | 2020 | 1.3549 | 1.7892 | 6 | 6 |
| 50% | 30000 | 2020 | 1.4029 | 1.8190 | 6 | 7 |
| 80% | 10000 | 2020 | 1.5957 | 2.2503 | 10 | 10 |
| 80% | 15000 | 2020 | 1.6395 | 2.2631 | 10 | 10 |
| 80% | 20000 | 2020 | 1.6917 | 2.2827 | 10 | 10 |
| 80% | 25000 | 2020 | 1.7501 | 2.3090 | 10 | 10 |
| 80% | 30000 | 2020 | 1.8136 | 2.3414 | 10 | 10 |
<!-- BOOK_MUSIC_K_SWEEP_END -->

## 图书到音乐 EMCDR 从头重跑

按用户要求，单独从随机初始化重跑 MF 图书到音乐的 EMCDR，
覆盖 20%、50%、80% 三个测试用户比例，各用 seed=2020 运行一次。
不加载此前的预训练模型或原型。已有数据划分仅在 SHA256 校验后复用。

执行源码的前置训练顺序：目标域 MF 10 个 epoch，源域 MF 10 个 epoch，
EMCDR 映射 10 个 epoch。调用现有 `Run.TgtOnly()` 和 `Run.EMCDR()`；
训练 batch 分别为 512、512、128，评估 batch=256。
采用 Adam、学习率 0.01、weight_decay=0、10 维嵌入及无偏置 10×10 映射。
不启用梯度累积，不运行模型反演、PFCDR 或 K 扫描。
论文提及预训练学习率搜索，本次使用公开源码默认 0.01，
不将其表述为已核实的作者逐任务最优学习率。

参数：[emcdr_book_music_fresh_plan.json](experiments/emcdr_book_music_fresh_plan.json)。
执行器：`experiments/run_emcdr_book_music_fresh.py run`。
各比例单独保存新模型、逐 epoch 指标和独立最低 MAE/RMSE，
结果写入 `results/emcdr_book_music_fresh/` 并自动 Git 提交推送。


### 图书到音乐 EMCDR 从头重跑结果

<!-- EMCDR_BOOK_MUSIC_FRESH_START -->
| 测试比例 | seed | 新跑 EMCDR MAE | 新跑 EMCDR RMSE | 论文 MAE | 论文 RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 2020 | 1.7241 | 2.0330 | 1.3524 | 1.6737 |
| 50% | 2020 | 2.0009 | 2.3191 | 1.4723 | 1.8000 |
| 80% | 2020 | 2.4670 | 2.8035 | 1.7191 | 2.1119 |
<!-- EMCDR_BOOK_MUSIC_FRESH_END -->

## 图书到音乐 EMCDR 学习率扫描

按用户要求，比较论文中的学习率候选 0.001、0.005、0.01、0.02、0.1，
覆盖 20%、50%、80% 三个测试用户比例，固定 seed=2020。
沿用公开源码共享 `lr` 的行为：同一个候选值用于目标域 MF、源域 MF 和 EMCDR 映射的 Adam。
其他设置保持不变：各阶段 10 个 epoch，训练 batch=512/512/128、测试 batch=256、
emb_dim=10、weight_decay=0，不启用梯度累积，不运行 PFCDR 或模型反演。

0.01 的三组刚已从随机初始化完成，本轮明确标注为“已完成的从头训练对照”纳入。
其余四个学习率各新训练三个比例，均从随机初始化开始，不加载旧模型。
因此包含 12 次新训练与 3 组已有对照，共 15 组比较，每组只有一次 seed=2020 运行。
每组分别取映射训练 10 个 epoch 内最低 MAE 和最低 RMSE，不进行多次运行平均。
保留全部结果与失败项，按最低实际误差标出各比例的候选结果，
不按与论文数值的接近程度选择试验。

计划：[emcdr_book_music_lr_grid_plan.json](experiments/emcdr_book_music_lr_grid_plan.json)。
执行入口：`experiments/run_emcdr_book_music_lr_grid.py run`。
每组开始前提交实际参数，完成后提交逐 epoch 记录和汇总，结果目录为
`results/emcdr_book_music_lr_grid/`。每个试验记录来源是本轮新训练还是已有对照。


### 图书到音乐 EMCDR 学习率扫描实际结果

<!-- EMCDR_BOOK_MUSIC_LR_GRID_START -->
| 测试比例 | 学习率 | MAE | RMSE | 与论文 MAE 差值 | 与论文 RMSE 差值 | 来源 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20% | 0.001 | 4.2940 | 4.4148 | +2.9416 | +2.7411 | 本轮从头训练 |
| 20% | 0.005 | 2.3728 | 2.6723 | +1.0204 | +0.9986 | 本轮从头训练 |
| 20% | 0.01 | 1.7241 | 2.0330 | +0.3717 | +0.3593 | 已完成的从头训练对照 |
| 20% | 0.02 | 1.3413 | 1.6643 | -0.0111 | -0.0094 | 本轮从头训练 |
| 20% | 0.1 | 4.5164 | 5.0902 | +3.1640 | +3.4165 | 本轮从头训练 |
| 50% | 0.001 | 4.3354 | 4.4509 | +2.8631 | +2.6509 | 本轮从头训练 |
| 50% | 0.005 | 2.7117 | 2.9914 | +1.2394 | +1.1914 | 本轮从头训练 |
| 50% | 0.01 | 2.0009 | 2.3191 | +0.5286 | +0.5191 | 已完成的从头训练对照 |
| 50% | 0.02 | 1.4258 | 1.7447 | -0.0465 | -0.0553 | 本轮从头训练 |
| 50% | 0.1 | 4.5338 | 5.1083 | +3.0615 | +3.3083 | 本轮从头训练 |
| 80% | 0.001 | 4.3286 | 4.4644 | +2.6095 | +2.3525 | 本轮从头训练 |
| 80% | 0.005 | 3.2421 | 3.4937 | +1.5230 | +1.3818 | 本轮从头训练 |
| 80% | 0.01 | 2.4670 | 2.8035 | +0.7479 | +0.6916 | 已完成的从头训练对照 |
| 80% | 0.02 | 1.5816 | 1.9603 | -0.1375 | -0.1516 | 本轮从头训练 |
| 80% | 0.1 | 4.5134 | 5.0728 | +2.7943 | +2.9609 | 本轮从头训练 |
<!-- EMCDR_BOOK_MUSIC_LR_GRID_END -->
