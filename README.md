# Reinforcement Learning Project: CliffWalking
This project implements a reinforcement learning environment called "CliffWalking," which is a variation of the classic Cliff Walking problem. The environment is designed as a subclass of CliffWalkingEnv from the Gym library. The project includes functionalities for policy evaluation and policy iteration within the Markov Decision Process (MDP) framework.

## MDP
MDP stands for Markov Decision Process. It is a mathematical framework used to model decision-making problems in situations where outcomes are partly random and partly under the control of a decision-maker.

In an MDP, the decision-making problem is represented as a tuple (S, A, P, R), 
where: 
- S is the set of possible states in the environment. 
- A is the set of possible actions that the decision-maker can take. 
- P is the state transition probability matrix, which defines the probability of transitioning from one state to another when a particular action is taken. 
- R is the reward function, which assigns a numerical reward to each state-action pair. 

The goal is to find an optimal policy that maximizes the expected cumulative reward over time.

## Policy Evaluation and Policy Iteration
The project implements policy evaluation and policy iteration algorithms for solving the CliffWalking environment. Policy evaluation estimates the value function for a given policy, while policy iteration alternates between policy evaluation and improvement to find the optimal policy in an MDP.

## Environment: CliffWalking
The implemented environment in this project called "CliffWalking" is a variation of the classic Cliff Walking problem. The environment is implemented as a subclass of CliffWalkingEnv from the gym library.


![](https://gymnasium.farama.org/_images/cliff_walking.gif)


### Attributes
- UP, RIGHT, DOWN, LEFT: Constants representing possible actions.
### Methods
  - __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs): Constructor method initializing the environment.
  - _calculate_transition_prob(self, current, delta): Helper method for calculating transition probabilities.
  - is_valid(self): Depth-first search (DFS) method to check for a valid path.
  - step(self, action): Overrides the step method for taking actions and returning state, reward, and termination status.
  - _render_gui(self, mode): Method for rendering the environment using the pygame library.

## How to Run 
1. Clone the Repository:
```bash
https://github.com/SheidaAbedpour/MDP-CliffWalking.git
```
2. Install Dependencies:
```bash
pip install -r requirement.txt
```
3. Run project:
```bash
python main.py
```
4. View the results, including the optimal policy and corresponding values.

## Acknowledgments
This project is based on the [CliffWalking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) environment from the Gym library. The project structure and documentation follow best practices and guidelines.
