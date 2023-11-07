import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import numpy as np


# a named tuple representing a single transition in our environment. 
# It essentially maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# a cyclic buffer of bounded size that holds the transitions observed recently. 
# It also implements a .sample() method for selecting a random batch of transitions for training.
class ReplayMemory(object):

    def __init__(self, capacity):
        # double ended queue
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    """The model has an output for each possible action e.g. for Cart-Pole we have 
    Q(s, left) and Q(s, right), where s is the input to the network and is drawn from
    the observation space.
    i.e. it predicts the expected return of taking each action given the current input
    
    Args:

        n_observations (int) : Size of the observation space e.g. for Cart-Pole this is 4:
            Cart position, Cart velocity, Pole angle, Pole angular velocity - meaning there
            are 4 values defining the state!
        n_actions (int) : Number of actions you can take e.g. for Cart-Pole this is 2: there
            are two possible actions: move left or move right
        """

    def __init__(self, observation_size, n_actions):
        super(DQN, self).__init__()
        self.layer_0 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.layer1 = nn.Linear(1051648, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer_0(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# Our aim will be to train a policy that tries to maximize the discounted, 
# cumulative reward at time t_0 = (sum from t=t_0 to t=inf; reward_at_time_t * discount_factor^(t-t_0)) 
# The discount, γ, should be a constant between 0 and 1 that ensures the sum converges. 
# A lower γ makes rewards from the uncertain far future less important for our agent than the ones in the near future that it can be fairly confident about. 
# It also encourages agents to collect reward closer in time than equivalent rewards that are temporally far away in the future.

# All below assumes transition from s -> s'

# The main idea behind Q-learning is that if we had a function Q∗:State×Action→R, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards:
# π∗(s)=argmax_{a}​ Q∗(s,a)

# However, we don’t know everything about the world, so we don’t have access to Q∗. 
# But, since neural networks are universal function approximators, we can simply create one and train it to resemble Q∗

# For our training update rule, we’ll use a fact that every Q function for some policy obeys the Bellman equation:
# Q^{π}(s,a)= r + γ*Q^{π}(s′,π(s′)) [1]
# 
# The difference between the two sides of the equality is known as the temporal difference error, 
# δ=Q(s,a)−(r+γ*max_{a}′​Q(s′,a)) [2]
# 
# Note the difference between equations [1] and [2] - we have a max because the Bellman equation
# says that the value of taking action a in state s = reward in state s + discounted reward for 
# being in state s' with the same policy  

# To minimize this error, we will use the Huber loss. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large 
# this makes it more robust to outliers when the estimates of Q are very noisy. 
# We calculate this over a batch of # transitions, B, sampled from the replay memory:
# L=(1/∣B∣)*∑_{(s,a,s′,r) ∈ B} L(δ)
# where L(δ)={.5*δ^{2} for ∣δ∣≤1 AND ∣δ∣−0.5 otherwise.

# Select action tries to balance exploration and exploitation using eps value
# If the random value is above the threshold it 'exploits' : the optimal action is chosen for the state
# Otherwise it 'explores': a random action is chosen
def select_action(steps_done, 
                  eps_end, 
                  eps_start, 
                  eps_decay, 
                  policy_net,
                  env,
                  device, 
                  state):
    
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done


def plot_durations(episode_durations, is_ipython, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model(memory, 
                   batch_size, 
                   gamma, 
                   optimizer, 
                   device, 
                   policy_net, 
                   target_net):

    # make sure can select batches from the memory i.e. batch_size < memory size
    if len(memory) < batch_size:
        return
    # randomly sample some transitions
    # By sampling from it randomly, the transitions that build up a batch are decorrelated. 
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    transitions = memory.sample(batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # this mask just allows us to select the non final states later
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # this actually lists the following states
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net or random
    # i.e. action batch is 0 or 1 or ... length_of_action space
    # dictating the action taken in state s - not necessarily the best action
    # just the action taken
    # state action values then gives the values of each of these actions
    # in the particular states
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # The target net tries to estimate the value of the following state, it is set up the same
    # as the policy net but they do not use the policy net for 'stability'
    # V(s_{t}+1​)=max_{a}​Q(s_{t+1}​,a)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the `optimizer
config = {
    'batch_size': 128,
    'gamma': 0.99,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 1000,
    'tau': 0.005,
    'lr': 1e-4,
}

# Training procedure

# We define a memory, policy net and target net
# For given number of episodes and steps within these episodes
# We select an action - either randomly or using the best action as predicted by 
# the policy network
# We note the reward and new state
# We add this transition (state, action, new state, reward) to the memory
#   Note that in memory we are not storing anything that we are predicting
#   only things given to us by the environment
# We then move to the next state
# We then optmimise the policy net:
#   Randomly sample transitions from memory
#   Use the policy net to calculate the Q value of the state for all possible actions
#   We then use the target network to calculate the Q value (i.e. best reward of all given actions)
#       for the following state (S') and then calculate the value of the original state using
#       bellman's equation: S = S' * gamma + r
#   We then calculate the difference between these two 
#  

def main(config):

    # Make the environment
    env = gym.make("ALE/Breakout-v5", render_mode = 'human')

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}")

    # get info about the state
    state, info = env.reset()
    input('stop')
    observation_size = state.shape
    n_actions = env.action_space.n
    print(f'The size of the observation space {observation_size}')
    print(f'The size of the action space is {n_actions}')

    # set up configuration
    batch_size = config['batch_size']
    gamma = config['gamma']
    eps_start = config['eps_start']
    eps_end = config['eps_end']
    eps_decay = config['eps_decay']
    tau = config['tau']
    lr = config['lr']

    # initialise networks and optimizer
    policy_net = DQN(observation_size, n_actions).to(device)
    target_net = DQN(observation_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

    # set up training
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_durations = []

    # an episode is defined as one complete running
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    losses = {i:[] for i in range(num_episodes)}

    for i_episode in range(num_episodes):
        
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.permute(state, (0, 3, 1, 2))

        # iterate through the number of states within an episode
        for t in count():
            
            # select action either randomly or using policy net

            action, steps_done = select_action(steps_done, 
                                   eps_end, 
                                   eps_start, 
                                   eps_decay, 
                                   policy_net, 
                                   env,  
                                   device, 
                                   state)
            # perform the action 
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = np.transpose(observation, (2, 0, 1))
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            #print('next state shape pre', next_state.shape, flush=True)
            #print('state shape pre', state.shape, flush=True)

            #state = torch.permute(state, (0,3,1,2))
            #next_state = torch.permute(next_state, (0,3,1,2))
            #print('state shape', state.shape, flush=True)
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(memory, 
                           batch_size, 
                           gamma, 
                           optimizer, 
                           device, 
                           policy_net, 
                           target_net)

            losses[i_episode].append(loss)

            # Soft update of the target network's weights
            # θ′ ← τ * θ + (1 −τ ) * θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                #plot_durations(episode_durations, is_ipython, show_result=False)
                break

    print('Complete')
    #plot_durations(episode_durations, is_ipython, show_result=False)
    plt.ioff()
    plt.show()

    print(losses[0])
    print(losses[49])



    

if __name__ == "__main__":
    main(config)