## Evolutionary Retrosynthetic Route Planning
This paper has been accept by IEEE Computational Intelligence Magazine. To learn more, see the [EvoRRP paper](https://ieeexplore.ieee.org/document/10595522).

### Quickstart
```shell
git clone https://github.com/ilog-ecnu/EvoRRP
cd EvoRRP
```

### Installation
```shell
conda env create -f environment.yml
conda activate evorrp
```

### Prepare Single-step Model
Use your own single-step retrosynthetic model to repalce `from single_step.infer import SingleInference` in file `algorithm/CUSOP.py`.

### Start Searching
```shell
python problem.py
```
If you want to save the result locally, execute the following command:
```shell 
python problem.py > log/log.out
```

## Citing EvoRRP
Please use the following bibtex entry to cite EvoRRP.
```
@article{zhang2024evolutionary,
  title={Evolutionary Retrosynthetic Route Planning [Research Frontier]},
  author={Zhang, Yan and He, Xiao and Gao, Shuanhu and Zhou, Aimin and Hao, Hao},
  journal={IEEE Computational Intelligence Magazine},
  volume={19},
  number={3},
  pages={58--72},
  year={2024},
  publisher={IEEE}
}
```