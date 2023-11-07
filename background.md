# Reinforcement learning

Summary of Artificial Intelligence: A modern approach, Russell and Norvig

## Learning from rewards

### Markov decision processes vs reinforcement learning

"Reinforcement learning is different from 'just solving an MDP' because the agent is not given the Markov Decision Process as a problem to solve, the agent is in the MDP. It may not know the transition model or the reward function and it has to act in order to learn more. Imagine playing a new game whose rules you don't know; after or hundred or so moves, the referees tells you 'you lose'."

### Sparse rewards

No instantaneous reward is given e.g. A game of chess you only get a reward once at the end of the game.

### Model based RL

"Agent uses a transition model of the environments to help interpret the reward signals and to make decisions about how to act. The model may be initially unknown, in which case the agent learns the model from observing the effects of its actions, or it may already be known - for example a chess program may know the rules of chess even if it does not know how to choose good moves"

"Often learns a utility function U(s), defined in terms of the sum of rewards from state s onward

### Model free RL

"Agent neither knows nor learns a transition model for the environment. Instead it learns a more direct representation"

1. Action-utility learning - Q-learning is the most common, the agent learns a Q-function $Q(s,a)$, which gives the total rewards from state s onwards if action a is taken. "Given a Q-function the agent can choose what to do in s by finding the action with the higest Q-value"
2. Policy search - The agent learns a policy $\Pi(s)$, that maps from states to actions

Note that U(s) = max_a Q(s,a) : Q-learning is model free 

## Passive RL

The agent's policy is fixed and the task is to learn the utilities of states and/or a model of the environment.

Start with simple case: 