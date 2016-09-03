import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    actions_values = {'forward': 10, 'left': 100, 'right': 1000, 'red': -1, 'green': 1, None: 1}

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.calculate_state(inputs, self.next_waypoint, deadline)
        print 'STATE:', self.state
        # TODO: Select action according to your policy
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def calculate_state(self, raw_state_dict, next_waypoint, deadline):
        """
        Generate unique values for each combination of the input.
        """
        light = self.actions_values[raw_state_dict['light']]
        oncoming = 1 if raw_state_dict['oncoming'] is None else self.actions_values[raw_state_dict['oncoming']]
        right = 5 if raw_state_dict['right'] is None else self.actions_values[raw_state_dict['right']] * 2
        left = 7 if raw_state_dict['left'] is None else self.actions_values[raw_state_dict['left']] * 3
        return light * (oncoming + right + left), self.actions_values[next_waypoint], deadline

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
