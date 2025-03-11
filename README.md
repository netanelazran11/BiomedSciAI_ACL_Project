# bmfm targets

Biomedical foundational models for target discovery

This repo contains the code for the [bmfm targets challenge](https://challenges.apps.res.ibm.com/challenges/6452?tab=details).
As of October 2023, it contains code for scRNA expression foundation models divided into the following sections:

* [bmfm_targets](./bmfm_targets/) a package implementing the pre-training of a foundation model on scRNA expression data, inspired by [scBERT](https://github.com/TencentAILabHealthcare/scBERT). Currently the pretraining works for BERT style models, over the PanglaoDB collection of scRNA datasets.
The [t5](./bmfm_targets/tasks/t5) sub-folder holds implementation of pre-training and fine-tunining tasks using the bmfm-core MAMMAL model, see it's [README](./bmfm-targets/tasks/t5/README.md) for more details.

* [benchmarks](./benchmarks/) downstream tasks that make use of foundation models for scRNA data

* [session_manager](./session_manager/) yamls and instructions for running jobs using the session manager from bmfm-core

* [aer2vec](./aer2vec/) SNP embedding code, see readme for more details.

Instructions for installing and running the pre-training and fine-tuning tasks are available in their respective directories.


## Environment

Code has been tested with Python 3.10 and 3.11 only.
Using a virtual environment for all commands in this guide is strongly recommended.
Both conda and vanilla venv environments are supported.

```sh
# create a conda enviornment "bmfm" with Python version 3.11
conda create -n bmfm python=3.11

# activate the enviornment before installing new packages
conda activate bmfm
```

## Installation

### For non-developers
The following command will install the repository as a Python package, and also attempt to install dependencies speficied in the setup.py file or the pyproject.toml. Note that the command does not clone the repositpry.

```sh
# assuming you have an SSH key set up on GitHub
# this
pip install "git+ssh://git@github.ibm.com/BiomedSciAI-Innersource/bmfm-targets.git@main"
```

### For developers


```sh
# Clone the cloned repository
git clone git@github.ibm.com:BiomedSciAI-Innersource/bmfm-targets.git

# Change directory to the root of the cloned repository
cd bmfm-targets

pip install --upgrade pre-commit

# install from the local directory
pip install -e ."[test]"   # ."[test,t5]" if t5 is needed; this installs many extra packages that are only used for t5

pre-commit install
```

To verify that your installation works, try running the tests:
```sh
pytest bmfm_targets
```
The tests may take a long time to execute (~35 min on some machines, ~15 min on Travis).
If the tests do not all pass, please open an issue in this repo.

## Contributing

Check [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Getting support

Check [SUPPORT.md](.github/SUPPORT.md).

## Credits
This project was created using https://github.ibm.com/BiomedSciAI-Innersource/python-blueprint.
