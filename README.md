
This readme is under construction... :)!

# Deep Q Learning with love(2d)

I had just finished the [machine learning course][9] given by professor Nando de Freitas, which he has kindly made available online, when I've seen the toy examples implemented by [Karpathy][1] in javascript. I was so amazed that I decided to implement the same examples using torch7 with lua.
Deep Learning and Reinforcement Learning are extremely exciting areas in Machine Learning, with a tremendous potential for applications in many other domains. I am particularly interested in applications in robot perception and manipulation. So, I have implemented these demos in order to get my head around the most recent developments in the area, giving me a good toolbox to think about my own research. I am also making it available to everyone else who wants to learn more about it :)!

This is a toy implementation of the Deep Q Network (DQN) algorithm as an attempt to reproduce the same results as Karpathy did in [his demos][1]. I have based this code on [Nando's lectures][9], specifically the last two lectures on deep reinforcement learning. I have also read the original approach [DQN code][10] from DeepMind folks. And finally, I have learnt many tricks by reading Kaparthy's [REINFORCEjs][11] original code.

The DQNAgent implemented in this project is a simplified version of the original DQN algorithm proposed by [Mnih][8] et al.

The big difference between Mnih's work and the toy examples implemented in this project is that we are not learning from pixel data, but we actually could if we wanted to! (It would just take a bit longer to train :X)

# Dependency list

* [Torch7][2]
* [nn][3]
* [love2d][4]
* [display][5] (for plotting)


# Running instructions

These instructions are targeted for OSX, but they should be very similar for Linux too! (Work in progress)

You will first need to install torch7 following the instructions on the [torch7 website][2].

If you have installed torch7 successfully, you now should download the Love game engine using this [link][6]. This application is responsible for running games that require the Love 2d game engine. You will also need to create an alias for the ```love``` application. Creating an alias you will allow you to call ```love``` from the terminal. The instructions for that can be found [here][7]. 

Alternatively, if you're on OSX, you can run the ```configure.bash``` script I have prepared:

```
source configure.bash
```

This will download the Love game engine and do all the set up I described for you. The Love game engine will be in the folder called ```dependencies```.


The last dependency you should install is ```display```. This is a lua rock for plotting charts and visualising them via a web browser. This way you can watch the performance of the DQN agent over time. You can install display following the instructions on [display][5] GitHub repository.


## Running the demo

Before actually running the demos, if you have installed the lua rock display and want to see the plots for the temporal difference error (TD) and average reward over time, you will need to run the plotting server first. You can open a separate terminal and from the main directory of the project you can run:

```
source plotting/start_server.bash
```

This will start a server and allow you to see the plots on your browser when running the demos using your localhost address http://127.0.0.1:8000.

To run the pre-trained demos you simply open a new terminal, and in the root folder of the project you can source the run scripts:

```
source run_puckworld_pretrained.bash
```

or

```
source run_waterworld_pretrained.bash
```

If you want to train your own agents, you can instead run: 

```
source run_puckworld.bash
```

or for the water world

```
source run_water_world.bash
```

You should be able to watch the agent playing around on your screen now. And also watch its performance on your browser as below.

![Demo setup](https://github.com/eaa3/DeepQLearning/raw/master/gifs/plotting.gif)


Additional features:

You choose the speed the simulation runs by pressing the keys from **1** (normal speed) to **7** (extremely fast), essentially changing the number of time step updates per iteration.

You can also pause learning by pressing **a**, or enable learning again by pressing **l**. 

Once you've decided your agent is good enough, you can save it by pressing **s**. This will serialise the agent's Q network named as a file qnet.7z, which you can load by modifying the provided scripts. Feel free for contributing with your own improvements and/or pre-trained models! Pull requests are welcome!

To pause the simulation you can press **p**. You can try to control the agent after pressing pause to have a feeling what it is like to be the agent! To control the agent you need to use the arrows on your keyboard.You can do experiments to compare your performance plots with the ones of your best-trained agent! Can you beat it? 

## Demos implemented so far

### PuckWorld

#### Description

This is a reproduction of the PuckWorld demo implemented by [Karpathy][1].

* **State space**: at each time step, the agent has access to a continuous state space represented by the vector **s**=**(px,py,vx,vy,fpx,fpy,epx,epy)**. The first two values **(px,py)** are the position of the agent, followed by its velocity **(vx,vy)**, the position of the green target **(fpx,fpy)**, and the enemy **(epx,epy)**.
* **Action space**: there are **5** possible actions that the agent can choose from at each time step. The agent can apply a fixed force to the left, right, up, down, or not apply a force at all.
* **Reward function**: the agent is punished with negative reward the farther away it is from the green target. It also receives extra punishment for entering the radius of effect of enemy in red. This makes the agent learn to avoid the red enemy, while at the same time trying to keep as close as possible to the green target.


![Puck World](https://github.com/eaa3/DeepQLearning/raw/master/gifs/puckworld_short.gif)

### WaterWorld

This is a reproduction of the WaterWorld toy example implemented by [Karpathy][1].

In this demo, the agent has to avoid the green moving circles (poisonous food) and eat as many red circles (good food) as it can.

#### Description

* **State space**: at each time step the agent can sense the environment via 30 antennas or range sensors that can measure the distance **d**, velocity **(fvx,fvy)** and type **(wall,green or red circle)** of the object up to a certain limit distance, as depicted below. In addition to that, the agent has access to its own velocity **(vx,vy)**. Since I am representing the type of sensed object as a [one-hot][12] vector, this gives a total of 30x6 + 2 = 182-dimensional vector representing our state space at each time step.
* **Action space**: there are **4** possible actions that the agent can choose from at each time step. The agent can apply a fixed force to the left, right, up or down.
* **Reward function**: the agent receives a positive reward when it touches a red circle, and a negative reward when it touches a green circle.

![Water World](https://github.com/eaa3/DeepQLearning/raw/master/gifs/waterworld_short.gif)


### Coming soon


* wait for it... :D

# License

MIT.


[1]: http://cs.stanford.edu/people/karpathy/reinforcejs/index.html
[2]: http://torch.ch/docs/getting-started.html#_
[3]: https://github.com/torch/nn
[4]: https://love2d.org/wiki/Main_Page
[5]: https://github.com/szym/display
[6]: https://love2d.org
[7]: https://love2d.org/wiki/Getting_Started
[8]: https://www.cs.toronto.edu/%7Evmnih/docs/dqn.pdf
[9]: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
[10]: https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
[11]: https://github.com/karpathy/reinforcejs
[12]: https://en.wikipedia.org/wiki/One-hot
