[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135608-be87357e-7d12-11e8-8eca-e6d5fabdba6b.gif "Trained Agent"


# Actor-Critic Methods

### Instructions

Open `DDPG.ipynb` to see an implementation of DDPG with OpenAI Gym's BipedalWalker environment.

### Results

![Trained Agent][image1]

### Algorithm Explaination
1. There are four netowrk needed to train. actor_target, actor_local, critic_local and critic_target
2. ReplayBuffer is a deque, which is predefined the buffer size and record latest Experience(state, action, reward, new_state, done).
3. Ornstein_Uhlenbeck Noise is added to action, whenever model evaluate the explict action.
4. actor_local and critic_local is updated everytime, actor_target and critic_target is updated through copying the parameters of local network  

**Training Process**
  Memory(ReplayBuffer) over batch size, start train Actors and Critics
  update once:
  update critic:
  1. get the Experience(`states`, `actions`, `rewards`, `next_states`, `dones`) with batch_size
  2. predict the `actions_next` using actor_target(`next_states`)` 
  3. TD estimation using critic_target, `Q_targets_next = critic_target(next_states, actions_next)` ,  `Q_targets = rewards + (gamma * Q_targets_next)`. `Q_target` is the estimation of Qvalue at current_state
  4. `Q_expected` is estimated by critic_local network.
  5. MES loss (`Q_expected`,`Q_target`), update critic_local. 
  update actor:
  1. `action_predict` estimated by actor_local
  2. `actor_loss = -critic_local(states, actions_pred).mean()` using the mean of value function to represent the loss of actor_local network
  soft update from local to target:
  `θ_target = τ*θ_local + (1 - τ)*θ_target`
