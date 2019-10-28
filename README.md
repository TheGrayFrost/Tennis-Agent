
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project: Collaboration and Competition

### Introduction

This project is my solution for the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

The Unity tennis environment has two agents bouncing a ball back and forth over a net using rackets. A reward of +0.1 is recieved if you hit the ball to the other side. If the ball falls or goes out of bounds, a reward of -0.01 is recieved. So, the goal is to keep hitting the ball over as long as you can.

The observation space has 8 variables for the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Getting Started

#### 1. You need to setup the environment before you can play with the code.
I coded everything on MacOS, but these instructions should work for Windows and Linux as well.

1. Create a virtual enviroment with python 3.6 and activate. All equirements will be installed in this enviroment.
2. Download the repo of Unity ML-Agents Toolkit at [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and follow the [install instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md)
3. The project requirements additionally include [Pytorch](https://pytorch.org/). A simple `pip install torch` would suffice.
4. Create an IPython kernel for the drlnd environment.
    
    ```
    python -m ipykernel install --user --name env_name --display-name "kernel_name"
    ```
    
#### 2. Clone this GitHub repository onto your machine. 

#### 3. Download the unity environment from one of the links below.  You need only select the environment that matches your operating system:

  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

#### 3. Open your Jupyter Notebook, open the Tennis.ipynb file and select the apropiate kernel (kernel_name set in step 1).

#### 4. Follow the instructions in `Tennis.ipynb` to train the Agent or view the pretrained agent interact with the enviroment

### Running the Code

There are 3 parts to the code:

#### 1. Exploring the enviroment and basic simulation using a random behavior agent. 
  These cells are described in markdowns.
   * Start the Environment
   * Examine the State and Action Spaces
   * Take Random Actions in the Environment

#### 2. Training the DDPG Agent. 
  Corresponds to the markdown **Solve: Training the Agent with DDPG**
   <br/> The agent stops when it reaches target average score of 0.5 over 100 episodes.
   
#### 3. Viewing the Trained Agent. 
  Corresponds to the markdown **Watch a Smart Agent!**.
   <br/> We load the trained model weights from the Saved Weights folder.
   <br/> To view the trained agent without going through the hassle of training, simply run all the cells in the first and the third  part.