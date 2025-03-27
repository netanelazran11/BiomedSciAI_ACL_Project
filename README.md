# biomed multi omics

Biomedical foundational models for omics data

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
pip install "git@github.com:BiomedSciAI/biomed-multi-omic.git"
```

### For developers


```sh
# Clone the cloned repository
git clone git@github.com:BiomedSciAI/biomed-multi-omic.git

# Change directory to the root of the cloned repository
cd biomed-multi-omics

pip install --upgrade pre-commit

# install from the local directory
pip install -e ."[test]"  

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
This project was created in IBM Research
