# Neural Architecture Constructor

IMPORTANT: This version of the tool is no longer maintained and is
here for archival. A newer version of the tool is available here: 
https://github.com/ciscoai/amla . If you need an AutoML tool
please consider cloning/forking/contributing to that.
 
The Neural Architecture Constructor (NAC) is a tool for automated construction
of neural networks.
It is used to construct neural networks based on an iterative sequence of
training, evaluation, metric extraction and network restructuring.
The tool currently supports neural network construction based on the algorithms 
described in the paper Neural Architecture Construction using EnvelopeNets 
(https://arxiv.org/abs/1803.06744).
 
Portions of the code are derived from code from the TensorFlow CIFAR-10 
tutorial model: 
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

If you publish work based on the Neural Architecture Constructor, please cite
https://arxiv.org/abs/1803.06744v2

Authors: 
Utham Kamath utham at sigfind.com
Abhishek Singh abhishs8 at cisco.com
Debo Dutta dedutta at cisco.com

## Prerequisites: 

- Follow instructions here to install TensorFlow in a virtualenv on.for GPU/CPU:
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv
- Alternatively, use an AWS DeepLearning AMI on an AWS GPU instance:
http://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/

## Install
```
    git clone https://github.com/CiscoAI/envelopenets
```
## Run
```
    cd nac
    #Edit nac_run.py to run the construction file that you wish to run
    #The file is currently set to run a construction run, followed by a few evaluation runs
    python nac_run.py
```

## Analyze
```
    cd nac/results
    tensorboard --logdir=.
    cd nac/plot
    #Copy the results files from the run/results directory to the plots directory
    python nac_accplot.py
    python nac_varianceplot.py
    python nac_structureplot.py
```
