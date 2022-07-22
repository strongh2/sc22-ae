### !!! How to reproduce the experiment in the submitted paper?
####  Choice-1: A jupyter notebook prepared.
One notebook runs on a virtual machine with a 32GB V100.  The runtime environment for this AE has been ready.  We recommend this choice to reviewers since the testing cases, launch scripts and descriptions have also been listed in this notebook.   <br>
**jupyter link**:  [http://47.111.26.83:8888/notebooks/sc22ae.ipynb](http://47.111.26.83:8888/notebooks/sc22ae.ipynb) <br>
**password**: see it in the ` Stage 1: Artifact Description` in the [sc submission system](https://submissions.supercomputing.org/) <br>

####  Choice-2: The docker image.
If having a machine with GPU(s),  reviewers can deploy the runtime environment via this docker image.  <br>
Please note to change the values of script arguments  and re-install the torch if the CUDA driver version is lower than 11.4, also highlighted in the next content. <br>
The following content describes the steps on how to use this docker image to reproduce experiments.

----

# StrongHold: Artifact Evaluation

This docker image helps reviewers reproduce the experiment results in the submitted paper for the AE committee. It has already included all the necessary datasets and baselines. Thus, most figures in the paper can be reproduced only following the next instructions. 

We provide a general description of how to launch a docker container using this image and how to run corresponding scripts to evaluate the system performance.

## Step1. Runtime Environment

### 1.1 download this docker image from the official hub website.

`docker pull strongh/sc22-ae:latest`

### 1.2  create a new docker container based on this image. 

`docker run -it -P -w /home/sys/STRONGHOLD --name=aetesting --network=host --gpus=all --ipc=host strongh/sc22-ae:latest /bin/bash`

### 1.3 check the runtime environment.

At this point, I believe the docker container is launched successfully, and the current terminal focus should be at the `/home/sys/STRONGHOLD` folder with `(py3.9.10)` virtual python environment, shown as

 `(py3.9.10) root@??:/home/sys/STRONGHOLD# `. 

If not at the `STRONGHOLD` folder, please use the `cd /home/sys/STRONGHOLD` command to change the current location.

If not at `(py3.9.10)` virtual python environment, please use `pyenv activate py3.9.10` to enter into `py3.9.10` virtual env.

### Note: if the host server's CUDA driver is lower than 11.4 or errors about torch version occur, please reinstall `torch`, `torchvision`, `apex` libraries using the following commands.
>
```bash
pip uninstall torch torchvision apex

STAGE_DIR=/home/sys/
cd ${STAGE_DIR}/pytorch && \
    pip install -r requirements.txt && \
    python setup.py clean && \
    python setup.py develop --cmake && \
    python setup.py install && \
cd ${STAGE_DIR}/vision && \
    python setup.py clean && \
    python setup.py install && \
cd ${STAGE_DIR}/apex && \
    pip install -r ./requirements_dev.txt && \
    pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' . && \
cd ${STAGE_DIR}/STRONGHOLD
```
## Step2. Evaluation.

To reuse the existing log files produced in the previous cases, we recommend you to run cases one by one, which reduces the total execution time to **about 5 hours**. 

- 2.1 CASE - The largest trainable model size (Figure 6a in Section VI.A)
- 2.2 CASE - Throughput  on the largest trainable model size supported by each baseline (Figure 7a in Section VI.B) 
- 2.3 CASE - Throughput on the largest trainable model size of Megatron-LM (Figure 8a in Section VI.B)
- 2.4 CASE - Nearly linear scaling as model size increases (Figure 8b in Section VI.B) 
- 2.5 CASE - Impact of working window size (Figure 9 in Section VI.C) 

**Log files are stored in `/home/sys/STRONGHOLD/results`** as a format of `log_[method]_l-[layers]_h-[hidden size]_bs-[BATCH_SIZE]_ws-[WINDOW_SIZE]_[date].txt`. We print the core content in the log files via `grep` and `awk` for you at the end of each execution.

**Launch script** `./examples/run.sh -m [method] -l [layers] -h [hidden size] -b [batch size] -w [window size]` accepts five arguements, where `[method]` takes the values of `megatron-lm`, `l2l`, `zero-offload`, `zero-infinity`, `stronghold` and `all`. Using all to automatically evaluate all approaches. Default values for `[layers]`, `[hidden size]`, `[batch size]`, `[window size]` are 16, 2048, 4 and 4, respectively.

### 2.1 The evaluation example on virtual machine with one 32G V100 GPU, 90GB CPU RAM and 12 CPU cores.
> **!!! Please change the corresponding arguments if the hardware differs from ours. !!!**

#### 2.1.1 CASE - The largest trainable model size (Figure 6a in Section VI.A)

In this case, we use GPT-like models to exploit each method's largest trainable model size. Model size changes via increasing/decreasing the number of transformer layers.

Here, we evaluate Megatron-LM, L2L, ZeRO-Offload, ZeRO-Infinity and STRONGHOLD on a virtual machine with one 32GB V100, 90GB CPU RAM and 12 CPU Cores to exploit their largest trainable model size and bottleneck. During this process, we configure the `Heads=16, Sequence Length=1024, Batch Size=4` in all GPT-like models and training setups.

The largest model sizes have been tested in this notebook, shown in the following table. Please run the following cells to reproduce it. Thanks.

| Methods | Largest Trainable Size | Layers | Hidden Size | Heads | Sequence Length | Batch Size |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Megatron-LM | **1.717 B**| **32** | 2048 | 16 | 1024 | 4 |
| L2L | **4.033 B**| **78** | 2048 | 16 | 1024 | 4 |
| ZeRO-Offload | **2.522 B**| **48** | 2048 | 16 | 1024 | 4 |
| ZeRO-Infinity | **2.522 B**| **48** | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **5.141 B**| **100** | 2048 | 16 | 1024 | 4 |

PS: `Errors about GPU/CPU OOM` might be represented as other information, such as 'can not create XXX'.


```
./examples/run.sh -m "megatron-lm" -l 32 -h 2048 && \
./examples/run.sh -m "l2l" -l 78 -h 2048 && \
./examples/run.sh -m "zero-offload" -l 48 -h 2048 && \
./examples/run.sh -m "zero-infinity" -l 48 -h 2048 && \
./examples/run.sh -m "stronghold" -l 100 -h 2048
```

> **`./examples/case1_extract.sh ` and `./examples/case1_draw.sh ` help you analysis log files and print the relevant information only.**


#### 2.1.2 CASE - Throughput on the largest trainable model size supported by each baseline (Figure 7a in Section VI.B) 

In this case, we use GPT-like models to exploit the largest trainable model size supported by each baseline and compare the performance against STRONGHOLD on each largest model size. Model size changes via increasing/decreasing the number of transformer layers.

Here, we evaluate (Megatron-LM, L2L, ZeRO-Offload, ZeRO-Infinity) v.s. STRONGHOLD on a virtual machine with one 32GB V100, 90GB CPU RAM and 12 CPU Cores. During this process, we configure the `Heads=16, Sequence Length=1024, Batch Size=4` in all GPT-like models and training setups.

The throughput has been tested in this notebook, shown in the following table. Please run the next cells to reproduce it. Thanks.

| Methods | Throughput | Trainable Size | Layers | Hidden Size | Heads | Sequence Length | Batch Size |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Megatron-LM | **0.7496** |1.717 B | 32 | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **0.6647** | 1.717 B| 32 | 2048 | 16 | 1024 | 4 |
|
| L2L | **0.0529** | 4.033 B| 78 | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **0.2271** | 4.033 B| 78 | 2048 | 16 | 1024 | 4 |
|
| ZeRO-Offload | **0.2523** |2.522 B | 48 | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **0.3999**| 2.522 B| 48 | 2048 | 16 | 1024 | 4 |
|
| ZeRO-Infinity | **0.2439** | 2.522 B| 48 | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **0.3999**| 2.522 B| 48 | 2048 | 16 | 1024 | 4 |

PS: Limitations of CPU cores and bandwidth in the virtual machine hurts the performance of STRONGHOLD a little.

```
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 48 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 78 -h 2048 -w 15
```

> **`./examples/case2_extract.sh ` and `./examples/case2_draw.sh ` help you analysis log files and print the relevant information only.**


#### 2.1.3 CASE - Throughput on the largest trainable model size of Megatron-LM (Figure 8a in Section VI.B) 

This case shows the throughput performance of running Megatron-LM, L2L, ZeRO-Offload, ZeRO-Infinity and STRONGHOLD, respectively, on a 1.717 B model that is the largest trainable model size supported by Megatron-LM. The evaluation is conducted on a virtual machine with one 32GB V100, 90GB CPU RAM and 12 CPU Cores.

The throughput results have been tested in this notebook, shown in the following table. Please run the following cells to reproduce it. Thanks.

| Methods | Throughput | Trainable Size | Layers | Hidden Size | Heads | Sequence Length | Batch Size |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Megatron-LM | **0.7496** | 1.717 B| 32 | 2048 | 16 | 1024 | 4 |
| L2L | **0.1729**| 1.717 B| 32 | 2048 | 16 | 1024 | 4 |
| ZeRO-Offload | **0.3711**| 1.717 B| 32 | 2048 | 16 | 1024 | 4 |
| ZeRO-Infinity | **0.3587** | 1.717 B| 32 | 2048 | 16 | 1024 | 4 |
| STRONGHOLD | **0.6647** | 1.717 B| 32 | 2048 | 16 | 1024 | 4 |

PS: Limitations of CPU cores and bandwidth in the virtual machine hurts the performance of STRONGHOLD a little.

```
./examples/run.sh -m "l2l" -l 32 -h 2048 && \
./examples/run.sh -m "zero-offload" -l 32 -h 2048 && \
./examples/run.sh -m "zero-infinity" -l 32 -h 2048
```

> **`./examples/case3_extract.sh ` and `./examples/case3_draw.sh ` help you analysis log files and print the relevant information only.**


#### 2.1.4 CASE - Nearly linear scaling as model size increases (Figure 8b in Section VI.B)

In this case, we evaluate the performance (elapsed time per iteration - ms) as the model size increases. Similar to previous cases, the model size changes via increasing/decreasing the number of transformer layers. 

You would see the `elapsed time per iteration` linearly rise with the number of transformer layers (representing model size), proving STRONGHOLD's scalability.

```
./examples/run.sh -m "stronghold" -l 92 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 64 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 56 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 40 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 24 -h 2048 -w 15 && \
./examples/run.sh -m "stronghold" -l 16 -h 2048 -w 15
```

> **`./examples/case4_extract.sh ` and `./examples/case4_draw.sh ` help you analysis log files and print the relevant information only.**


#### 2.1.5 CASE - Impact of working window size (Figure 9 in Section VI.C) 

Working window size affects the throughput. The larger window can better overlap GPU computation with data transfer, leading to higher training throughput. But, a larger window size means more GPU memory occupancy.

This case evaluates the impact of working window size for STRONGHOLD with 1.7B model. You will see that at the first stage, the larger window size can gain more benefits, while at the end of the stage, enlarging window size shows no influence because the current window size can hide the data transformation process.

PS: The bandwidth restriction in the virtual machine might slightly hurt the performance of STRONGHOLD.

```
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 2 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 4 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 6 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 8 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 10 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 12 && \
./examples/run.sh -m "stronghold" -l 32 -h 2048 -w 14
```

> **`./examples/case5_extract.sh ` and `./examples/case5_draw.sh ` help you analysis log files and print the relevant information only.**


