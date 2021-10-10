# -IEEE-TIM-2021-1-Shallow-CNN-for-HAR
[IEEE TIM 2021-1] Shallow Convolutional Neural Networks for Human Activity Recognition using Wearable Sensors
![Model](https://github.com/wenbohuang1002/-IEEE-TIM-2021-1-Shallow-CNN-for-HAR/blob/main/Images/Arch.png)
All of datasets we use in this paper can be download from Internet and you can find we how to process data in this paper.  
This is my first time to open source, so there maybe some problems in my codes and I will improve this project in the near feature.  
Thanks!
## Requirements
● Python3  
● PyTorch (My version 1.9.0+cu111, please choose compatibility with your computer)  
● Scikit-learn  
● Numpy
## How to train
### UCI-HAR dataset
Get UCI dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets
```
# Baseline (3-layer CNN) for UCI-HAR
$ python Net_UCI_B.py
# 6-layer CNN for UCI-HAR
$ python Net_UCI_B1.py
# C3 for UCI-HAR
$ python Net_UCI_C3.py
```
### OPPORTUNITY dataset
```
# Baseline (3-layer CNN) for OPPORTUNITY
$ python Net_Opportunity_B.py
# 6-layer CNN for OPPORTUNITY
$ python Net_Opportunity_B1.py
# C3 for OPPORTUNITY
$ python Net_Opportunity_C3.py
```
### PAMAP2 dataset
```
# Baseline (3-layer CNN) for PAMAP2
$ python Net_pamap2_B.py
# 6-layer CNN for PAMAP2
$ python Net_pamap2_B1.py
# C3 for PAMAP2
$ python Net_pamap2_C3.py
```
### UniMiB-SHAR dataset
```
# Baseline (3-layer CNN) for UniMiB-SHAR
$ python Net_unimib_B.py
# 6-layer CNN for UniMiB-SHAR
$ python Net_unimib_B1.py
# C3 for UniMiB-SHAR
$ python Net_unimib_C3.py
```

## Citation
If you find Shallow CNN for HAR useful in your research, please consider citing.
```
@article{huang2021shallow,
  title={Shallow Convolutional Neural Networks for Human Activity Recognition Using Wearable Sensors},
  author={Huang, Wenbo and Zhang, Lei and Gao, Wenbin and Min, Fuhong and He, Jun},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={70},
  pages={1--11},
  year={2021},
  publisher={IEEE}
}
```
