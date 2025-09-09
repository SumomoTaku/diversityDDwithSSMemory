# Diversity-Driven Generative Dataset Distillation Based on Diffusion Model with Self-Adaptive Memory
The code used in the following paper:<br>
[Diversity-Driven Generative Dataset Distillation Based on Diffusion Model with Self-Adaptive Memory](https://arxiv.org/abs/2507.03331)

The code is modified from that of **[Minimax](https://github.com/vimar-gu/MinimaxDiffusion)**, following the paper **[Efficient Dataset Distillation via Minimax Diffusion](https://arxiv.org/abs/2311.15529).**<br>
In addition to the new parts mentioned in our paper, the modification also includes the following points:<br>
1. The Python libraries are updated to a newer version. (Especially for the PyTorch family, which usually raises deprecated warnings.)
2. Some codes have been rewritten (based on personal preference) (for myself to better understand the works).
3. The arguments of the 3 parts are integrated into 1 file (although it still looks chaotic).
4. Some arguments and statements are removed as being considered unrelated to the current task (from the perspective of an AI beginner).

If you find some detailed parameters hard to regulate, please refer to the original code listed above.<br>
<br>
Code rewriting will be continuing ...

## How to Use
1. Set the virtual environment with conda<br>
```
git clone https://github.com/SumomoTaku/diversityDDwithSSMemory.git
cd diversityDDwithSSMemory
conda env create -f environment.yml
conda activate divMem
```
2. Run the demo script. <br>
The script contains all 3 parts: "fine-tuning the DiT model", "obtaining the distilled dataset", and "training the downstream model".<br>
For subsets of ImageNet, lists of classes should be stored in a file, following the format of "misc/class_indices.txt".<br>
It should be passed to the parameter '--select_list' during training.<br>
(Only experiments on ImageNet have been conducted with this code.)<br>
```
cd scripts
sh run.sh
```
