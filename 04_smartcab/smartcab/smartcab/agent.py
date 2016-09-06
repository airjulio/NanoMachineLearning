from __future__ import division
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import pprint
import operator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.qtable = defaultdict(dict)
        self.epsilon = .4  # Randomness factor
        self.alpha = .4  # Learning rate.

    def reset(self, destination=None):
        # Prepare for a new trip; reset any variables here, if required
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.define_state(inputs, self.next_waypoint)

        # Select action according policy
        action = self.select_action()

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.update_qtable(action, reward)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
        #                                                                                             action,
        #                                                                                             reward)  # [debug]

    def select_action(self):
        if random.random() < self.epsilon or self.state not in self.qtable:
            action = random.choice(self.env.valid_actions)
        else:
            action = max(self.qtable[self.state], key=self.qtable[self.state].get)

        if self.state not in self.qtable:
            for a in self.env.valid_actions:  # init action:reward dict.
                self.qtable[self.state][a] = 0
        return action

    def update_qtable(self, action, reward):
        self.qtable[self.state][action] = \
            ((1 - self.alpha) * self.qtable[self.state].get(action, 0)) + (self.alpha * reward)
        # Decay alpha and epsilon until it reaches 0.1 and 0.0, respectively.
        self.alpha = max(0.001, self.alpha - 0.002)
        self.epsilon = max(0.0, self.epsilon - 0.002)

    @staticmethod
    def define_state(inputs, next_waypoint):
        return tuple(inputs.values()) + (next_waypoint, )


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    trials = 1000
    sim.run(n_trials=trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    for i, state in enumerate(a.qtable.items()):
        print i, state[0]
        pprint.pprint(state[1])
    print 'Success rate after {} trials: {}.'.format(trials, sim.success / trials)


def run_grid_search():
    """Run the agent for a finite number of trials."""

    alpha_params = [1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]
    epsilon_params = [1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]
    results = {}
    for alpha in alpha_params:
        for epsilon in epsilon_params:
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0.0,
                            display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False
            trials = 1000
            a.alpha = alpha
            a.epsilon = epsilon
            sim.run(n_trials=trials)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            # for i, state in enumerate(a.qtable.items()):
            #     print i, state[0]
            #     pprint.pprint(state[1])
            print 'Success rate after {} trials: {}. Initial alpha = {} and initial epsilon = {}' \
                .format(trials, sim.success / trials, alpha, epsilon)
            results['a:{} e:{}'.format(alpha, epsilon)] = sim.success / trials
    pprint.pprint(sorted(results.items(), key=operator.itemgetter(1)))


if __name__ == '__main__':
    run()
