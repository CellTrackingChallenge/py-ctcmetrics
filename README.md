[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Pylint](https://github.com/TimoK93/ctc-metrics/actions/workflows/pylint.yml/badge.svg)](https://github.com/TimoK93/ctc-metrics/actions/workflows/pylint.yml)
[![Python package](https://github.com/TimoK93/ctc-metrics/actions/workflows/python-package.yml/badge.svg)](https://github.com/TimoK93/ctc-metrics/actions/workflows/python-package.yml)

# CTC-Metrics
An **unofficial** python implementation of the metrics used in the 
[Cell-Tracking-Challenge](https://celltrackingchallenge.net/).

---

**WORK IN PROGRESS!**

PLEASE DO NOT USE! No Guarantee that it works as intended!

The first tested version will be published soon.

---

### Requirements

We tested the code with **Python 3.10**. Additional packages that will be 
installed automatically are listed in the [requirements.txt](requirements.txt).

## Installation

The package can be installed via pip:

```bash
pip install git+https://github.com/TimoK93/ctc-metrics.git
```

or from the source code:

```bash
git clone https://github.com/TimoK93/ctc-metrics
cd ctc-metrics
pip install .
```

## Usage


The package supports evaluation and validation of tracking results. The 
procedure can be shown for an example directory structure in the CTC-format
structured as follows:

```bash
ctc
├── train
│   ├── challenge_x
│   │   ├── 01_GT
│   │   ├── 01_RES
│   │   ├── 02_GT
│   │   ├── 02_RES
│   ├── challenge_y 
│   │   ├── 01_GT
│   │   ├── 01_RES
│   │   ├── 02_GT
│   │   ├── 02_RES
```


To validate if the sequence ```challenge_x/01``` is correctly formatted, run the 
command
```bash
ctc_validate --res "/ctc/train/challenge_x/01_RES"
```
Moreover, you can recursively validate the tracking results for all 
challenges/sequences in a directory by adding the flag ```-r```:
```bash
ctc_validate --res <path/to/res> -r
```
In this example, all four sequences will be validated.


To evaluate results against Ground Truth, similar commands can be used. 
To evaluate the results of the sequence ```challenge_x/01``` against the
corresponding ground truth, run the command
```bash 
ctc_evaluate --gt "/ctc/train/challenge_x/01_GT" --res "/ctc/train/challenge_x/01_RES"
```
or recursively for all sequences in a directory:
```bash
ctc_evaluate --gt "/ctc/train" --res "/ctc/train" -r
```
There are additional arguments that can be used to specify the evaluation. The
following table shows the available arguments:

| Argument    | Description                                         | Default |
|-------------|-----------------------------------------------------| --- |
| --gt        | Path to the ground truth directory                  | None |
| --res       | Path to the results directory                       | None |
| --recursive | Recursively evaluate all sequences in the directory | False |
| --csv-file  | Path to a csv file to save the results              | None |

Per default, all metrics are evaluated. Additional arguments to select a subset 
of specific metrics are:

| Argument | Description | 
| --- | --- |
| --valid | Check if the result has valid format | 
| --det | The DET detection metric |
| --seg | The SEG segmentation metric |
| --tra | The TRA tracking metric |
| --ct | The CT (complete tracks) metric |
| --tf | The TF (track fraction) metric |
| --bc | The BC(i) (branching correctness) metric |
| --cca | The CCA (cell cycle accuracy) metric |

---

To use the evaluation protocol in your python code, the code can be imported
as follows:

```python   
from ctc_metrics import evaluate_sequence, validate_sequence

# Validate the sequence
res = validate_sequence("/ctc/train/challenge_x/01_RES")
print(res["Valid"])

# Evaluate the sequence
res = evaluate_sequence("/ctc/train/challenge_x/01_RES", "/ctc/train/challenge_x/01_GT")
print(res["DET"])
print(res["SEG"])
print(res["TRA"])
...
    
```

## Contributing

Contributions are welcome! For bug reports or requests please
[submit an issue](www.github.com/TimoK93/ctc-metrics/issues). For new features
please [submit a pull request](www.github.com/TimoK93/ctc-metrics/pulls).

If you want to contribute, please check your code with pylint and the
pre-commit hooks before submitting a pull request:

```bash
pip install pre-commit, pylint
pre-commit run --all-files
```

## Acknowledgement and Citation

The code was developed by Timo Kaiser on behalf of the [Institute of Information
Processing](https://www.tnt.uni-hannover.de/) at the Leibniz University Hanover 
and in conjunction with the organizers of the
[Cell-Tracking-Challenge](https://celltrackingchallenge.net/).


If you use this code in your research, please cite the following paper:

```bibtex
@article{thecelltrackingchallenge,
    author = {Maška, Martin and Ulman, Vladimír and Delgado-Rodriguez, Pablo and Gómez de Mariscal, Estibaliz and Necasova, Tereza and Guerrero Peña, Fidel Alejandro and Ing Ren, Tsang and Meyerowitz, Elliot and Scherr, Tim and Löffler, Katharina and Mikut, Ralf and Guo, Tianqi and Wang, Yin and Allebach, Jan and Bao, Rina and Al-Shakarji, Noor and Rahmon, Gani and Toubal, Imad Eddine and Palaniappan, K. and Ortiz-de-Solorzano, Carlos},
    year = {2023},
    month = {05},
    pages = {1-11},
    title = {The Cell Tracking Challenge: 10 years of objective benchmarking},
    volume = {20},
    journal = {Nature Methods},
    doi = {10.1038/s41592-023-01879-y}
}
```

