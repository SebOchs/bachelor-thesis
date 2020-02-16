# Using Deep Learning Techniques for Automatic Short Answer Grading

Bachelor Thesis researching the robustness of deep learning (DL) models for automatic short answer 
grading (ASAG) against adversary examples  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The project was created on a PC with Linux Mint 19.2 as OS

### Installing

Create the two conda environments with the provided yaml files:

```
$ conda env create --file asag_environment.yaml
$ conda env create --file sear_environment.yaml
```

To preprocess the SemEval 2013 Short Answer Grading datasets, 
activate the ASAG environment and run the preprocessing script in bert

```
$ conda activate asag
$ cd bert
$ python preprocessing.py
```

Clone modified SEAR to project

```
git clone https://github.com/SebOchs/sears.git
```

Train ASAG model

```
todo
```

todo: show example


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Acknowledge Marcos SEAR code
* Acknowledge other help


