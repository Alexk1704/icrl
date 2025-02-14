import os
import sys
import morph
import pprint
import argparse

from utils import helpers


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_policy_1', help='custom_1 policy help')
parser.add_argument('--custom_2', type=str, default='default_policy_2', help='custom_2 policy help')


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    yield plot_q_value
    yield plot_td_error


# timeseries data by default
# depend on the given state (not the action)
def plot_q_value():
    labels, data = helpers.extract_data(['q_values'])
    helpers.generic_plot(data, labels)

    labels, data = helpers.extract_data(['online_q_values'])
    helpers.generic_plot(data, labels)

    labels, data = helpers.extract_data(['offline_q_values'])
    helpers.generic_plot(data, labels)

# timeseries data by default
# depend on the given state and the action
def plot_td_error():
    labels, data = helpers.extract_data(['td_error'])
    helpers.generic_plot(data, labels)

    labels, data = helpers.extract_data(['online_td_error'])
    helpers.generic_plot(data, labels)

    labels, data = helpers.extract_data(['offline_td_error'])
    helpers.generic_plot(data, labels)

# am ende nochmal künstliche mini-batch auf dem letzten stand überprüfen
# für q-values geht das, nur state vorgegeben und q-values erhalten
# für td-error muss eine action gewählt werden
# daher alle actions jeweils einmal auswählen, um rewards zu haben? (reicht ja einmal für alle tasks oder random walk => max exploration samples speichern)
# on-the-fly collecten ist zu umständlich => nur state vorgegeben und dann max/min/random action
'''
Q-Values Convergence (Evolution): Plotting the change in Q-values to check for convergence.
TD Error Histograms: Plotting histograms of temporal-difference (TD) errors to understand how well the agent is predicting future rewards.

Value Function Heatmap:
    Create a heatmap of the estimated value function for different states in the environment.
    This illustrates how the agent assigns values to different states and how it evolves during training.
(Quantisiert)

Uncertainty Visualization:
    Visualize the agent's uncertainty in its value estimates or policy.
    This can be done using error-bars or heatmaps, indicating areas where the agent is less confident. (state-action space)
    Create a heatmap of the environment where each cell represents a state.
    Color-code the cells based on the level of uncertainty associated with each state.

Contour plot: A contour plot of the Q-values learned by the agent in the state-action space.
Contour plot: A contour plot of the TD-error learned by the agent in the state-action space.

networkx mdp
Network graph: A network graph showing the connections between different states or actions based on the reward or state-action values. # geht immer, auch on-the-fly
Network graph: A network graph showing the connections between different states or actions based on their Q-values or state-action values. # geht nur am Ende, wenn konvergiert und nicht on-the-fly, muss explizit nochmal traversiert werden.
Network graph: A network graph showing the connections between different states or actions based on their TD-error or state-action values. # geht nur am Ende, wenn konvergiert und nicht on-the-fly, muss explizit nochmal traversiert werden.
'''
