# Pong-DQL
DQL Model that learns to beat Atari Pong using PyTorch, Tensorflow, Keras

![ataripong](https://github.com/harmanbrar7/Pong-Deep-Q-Learning/assets/89001739/de8696bc-3070-4c71-9c67-059c74aa8b98)


Uses OpenAI's Gym environment. https://github.com/openai/gym


```bash
#PyTorch
pip3 install torch torchvision
```
```bash
#OpenAI Gym
pip3 install gym atari-py
```
```bash
#OpenCV Python
pip3 install opencv-python
```
```bash
#PyTorch
pip3 install torch torchvision
```
```bash
#TensorboardX
pip3 install tensorboardX
``` 


##4 key files:
dqn_basic - main program
wrappers - gym enviro wrappres
utils - plotting
dqn_model - pytorch framework

To Run: 
dqn_basic.py 
*if using cuda* - then 'dqn_basic.py --cuda'

TensorBoardX Run:
'tensorboard --logdir runs'


demo.mp4 shows how the model learns to easily beat the game 

NOTE 1: Please note which versions of Keras, TensorFlow, PyTorch and Python you are using. 
TensorFlow 2.0 and above contains Keras, so run code accordingly. Not recommended for notebook environments, use GPU.

NOTE 2: Gym has now changed to Gymnasium so there may be subtle differences in version dependencies.
