# A Statistical Test for Probabilistic Fairness
Authors: Bahar Taskesen, Jose Blanchet, Daniel Kuhn, Viet-Anh Nguyen

ACM FACCT 2021

## Contents
- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Validation of the Hypothesis Test](#validation-of-the-hypothesis-test)
- [Numerical Experiment](#numerical-experiment)

## Quick Start
To install the required packages, use
```shell
pip install -r requirements.txt
```
The datasets are shared under 
Here `./data` is the path to the datasets.

## Introduction
Algorithms are now routinely used to make consequential decisions that affect human lives. Examples include college admissions, medical interventions or law enforcement. While algorithms empower us to harness all information hidden in vast amounts of data, they may inadvertently amplify existing biases in the available datasets. This concern has sparked increasing interest in fair machine learning, which aims to quantify and mitigate algorithmic discrimination. Indeed, machine learning models should undergo intensive tests to detect algorithmic biases before being deployed at scale. In this paper, we use ideas from the theory of optimal transport to propose a statistical hypothesis test for detecting unfair classifiers. Leveraging the geometry of the feature space, the test statistic quantifies the distance of the empirical distribution supported on the test samples to the manifold of distributions that render a pre-trained classifier fair. We develop a rigorous hypothesis testing mechanism for assessing the probabilistic fairness of any pre-trained logistic classifier, and we show both theoretically as well as empirically that the proposed test is asymptotically correct. In addition, the proposed framework offers interpretability by identifying the most favorable perturbation of the data so that the given classifier becomes fair.


## Validation of the Hypothesis Test
In Section 6 of our paper, we demonstrate that our proposed Wasserstein projection framework for statistical test of fairness is a valid, or asymptotically correct, test.
The data used to generate plots in Figure 2 is obtained by runing
```shell
python hyp_test_PEOPP.py 
```
Later on we use this data to generate the pdf and cdf plots in Figure 2 by running
```shell
python ./results/plot_results.py 
```
and the rejection percentage values noted in Table 1 are saved in the variable of `test_results`. 

## Numerical Experiment 

In Section 6, we further conduct an experiment with a Tikhonov regularized logistic regression classifier trained on a modern dataset, COMPAS.
We vary the value of regularization parameter and test the fairness of the logistic classifier.
Test statistic and accuracy of Tikhonov regularized logistic regression on test data with a predetermined rejection threshold.
```shell
python ./hypothesis_test_reg_logistic.py 
```
