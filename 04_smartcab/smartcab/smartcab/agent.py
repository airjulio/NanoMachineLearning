import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
from collections import defaultdict


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    actions_values = {'forward': 10, 'left': 100, 'right': 1000, 'red': -1, 'green': 1, None: 1}

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qtable = defaultdict(dict)
        self.gamma = 1.0  # Discount factor.
        self.alpha = .5  # Learning rate.

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = LearningAgent.calculate_state(inputs, self.next_waypoint)

        # Select action according to your policy
        if random.random() > self.gamma and self.state in self.qtable:
            action, _ = LearningAgent.get_best_action(self.qtable[self.state])
        else:
            action = random.choice(self.env.valid_actions)
            for a in self.env.valid_actions:  # init action:reward dict.
                self.qtable[self.state][a] = self.qtable[self.state].get(a, 1)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.qtable[self.state][action] = \
            ((1 - self.alpha) * self.qtable[self.state].get(action, 0)) + (self.alpha * reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    action,
                                                                                                    reward)  # [debug]
        # Decay alpha and epsilon until it reaches 0.1 and 0.0, respectively.
        self.alpha = max(0.1, self.alpha - 0.02)
        self.gamma = max(0.0, self.gamma - 0.02)

    @staticmethod
    def get_best_action(actions):
        best_action = None
        max_reward = -1000
        for action, reward in actions.items():
            if reward > max_reward:
                max_reward = reward
                best_action = action
        return best_action, max_reward

    @staticmethod
    def calculate_state(raw_state_dict, next_waypoint):
        return tuple(raw_state_dict.values()) + (next_waypoint,)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    # from pprint import pprint
    # for state, actions in a.qtable.items():
    #     print state
    #     pprint(actions)


if __name__ == '__main__':
    run()
