# Factorized Score Based Diffusion Model (FSDM)
FSDM is a way of factorizing the posterior score into its consituent prior and inidividual likelihood scores as described in the paper: "A generative foundation model for five-class sleep staging with arbitrary sensor input". This repository provides an example of the FSDM algorithm and how it can be leveraged in a simple classification example.

### Python required dependencies
- numpy
- torch
- matplotlib

### The example
In the example we have a classification problem consisting of 4 classes. Additionally, we are measuring 2 signals that provide insufficient information on their own to find the correct class. That is because signal 1 can only tell if the correct class is either ('0','1') or ('2','3'), while signals 2 cna tell if the correct class is either ('0','2') or ('1','3'). Only by combining the signals is full knowledge possible. We show in an example how seperate score models can be leveraged to solve this problem using the FSDM rule. To see the example, run:
```
sampling.py
```