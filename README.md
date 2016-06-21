
## Deep Q Learning with love(2d)

I was so amazed when I read the simple toy examples implemented by Karpathy [link][1] in javascript that I decided to implement the same examples using torch7 with lua.



This is a toy implementation of the Deep Q Network (DQN) algorithm as an attempt to reproduce the same results as Karpathy did in his demos. 

DQNAgent implements the DQN algorithm proposed by Mnih et al
https://www.cs.toronto.edu/%7Evmnih/docs/dqn.pdf

A big difference between Mnih's work and the toy examples implemented in this project is that we are not learning from pixel data. The word "Deep" in machine learning litterature stems from the fact that we can learn useful features to use directly from the raw input (e.g. images). Instead, we are specifying low level input features representing the state of underlying simulated environment, such as in [link][1]. However, this is actually enough to extract the power of neural networks as function approximators in the DQN algorithm.Since our examples in this project are simple, we can easily specify handmade features that capture well the state of the environment. Nonetheless, it is good for you to know that if you ever find yourself with no idea of what features to use, you can simply add more layers on your network, convolutional layers in the case of images, for example, which will naturally learn meaninful features in an end-to-end fashion.

## Dependency list

* Torch7 [link][2]
* nn [link][3]
* love2d [link][4]
* display (for plotting)[link][5] 

[1]: http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html
[2]: http://torch.ch
[3]: https://github.com/torch/nn
[4]: https://love2d.org/wiki/Main_Page
[5]: https://github.com/szym/display
[6]: https://love2d.org/wiki/Game_Distribution#Creating_a_MacOS_X_App
[7]: https://love2d.org/wiki/Getting_Started

## Running instructions for OSX

You will first need to install torch7 follwoing the instructions in [link][2].

If you have installed torch7 successfully, you now should install love using this link [link][6]. This application is responsible for running games written using love2d game engine. You will also need to create an alias for the love application. Creating an alias you will allow you to call love command from the terminal.

The last dependency you should install is display. This is a lua rock for plotting charts and visualising them via a webbrowser. This way you can watch the performance of the DQN angent over time. You can install display following the instructions on [link][5].


### Running the demo

To run the pretrained demos you simply need to source the run scripts provided in the main directory of the project. If you want to train your own agents, simply modify the provided scripts. Those scripts are simply passing an optional pretrained Q-Network. If you remove the pretrained network argument, the agent will learn from scratch.


## Demos implemented so far

### PuckWorld

### WaterWorld

## Coming soon

