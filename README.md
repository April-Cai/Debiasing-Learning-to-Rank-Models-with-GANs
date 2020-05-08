# Debiasing-Learning-to-Rank-Models-with-GANs
## Overview
This is an implementation of the paper "Debiasing Learning to Rank Models with Generative Adversarial Networks".

## Requirements
##### 1.environments
Python 3.6+ and Pytorch v1.2+ are needed.
##### 2.data
Download generated dataset of Yahoo Letor by [Ziniu Hu, et al.](https://github.com/acbull/Unbiased_LambdaMart) and put them under the path ./data/Yahoo.
##### 3.evaluation
Download [TREC](https://trec.nist.gov/trec_eval/) tools and put files under the path ./tools/trec_eval
## Example
```
cd main
sh test.sh  # test on sample data
sh run.sh  # test on yahoo data
```
The log file is saved under the path ./log/