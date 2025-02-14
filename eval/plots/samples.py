import os
import sys
import morph
import pprint
import argparse

from utils import helpers


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_samples_1', help='custom_1 samples help')
parser.add_argument('--custom_2', type=str, default='default_samples_2', help='custom_2 samples help')


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    yield plot_static
    yield plot_random
    yield plot_clock
    yield plot_state
    yield plot_action
    yield plot_reward
    yield plot_duration


# each entity itself
# special plots single entity
def plot_static():
    labels, data = helpers.extract_sample(['static'])
    helpers.generic_plot(data, labels)

def plot_random():
    labels, data = helpers.extract_sample(['random'])
    helpers.generic_plot(data, labels)

def plot_clock():
    labels, data = helpers.extract_sample(['clock'])
    helpers.generic_plot(data, labels)

def plot_state():
    labels, data = helpers.extract_sample(['state'])
    helpers.generic_plot(data, labels)

def plot_action():
    labels, data = helpers.extract_sample(['action'])
    helpers.generic_plot(data, labels)

def plot_reward():
    labels, data = helpers.extract_sample(['reward'])
    helpers.generic_plot(data, labels)

def plot_duration():
    labels, data = helpers.extract_sample(['duration'])
    helpers.generic_plot(data, labels)

# combine/compare multiple entities
# special plots multiple entities
def plot_episode_length():
    pass

def plot_episode_returns():
    pass

'''
State Visitation Frequency:
    Create a heatmap or histogram of the states visited by the agent during training.
    This highlights which states the agent frequently encounters and helps identify any bottlenecks or underexplored areas.

Action Pick Rate:
    Plot a histogram of the actions taken by the agent during training.
    This provides insights into the agent's action preferences and how they change over time.

Exploration-Exploitation Trade-off:
    Plot the balance between exploration (e.g., using entropy or the number of unique actions) and exploitation (e.g., cumulative reward) over episodes.

Vollständigkeit der Exploration
    Heatmap, wie Value Function oben, aber mit Häufigkeit des Eintreffens (2D-Hist) anstatt den Q-Values

Reward Heatmap
    Heatmap, wie Value Function, aber mit dem Reward für das State-Action Tuple anstatt den Q-Values

Episode Return over Time: Tracking the return of episodes over time to observe if the agent is learning to solve the task more rewardingly.
Episode Length Over Time: Tracking the length of episodes over time to observe if the agent is learning to solve the task more consistently.
Reward Heatmaps: For grid environments, showing a heatmap of collected rewards per state. (anstatt unsere Heatmap der Trajektorie)

State-to-state transitions (als animation?) -> aussagekräftiger über die estimated/approximated value-function
Action-to-action transitions (als animation?) -> aussagekräftiger über die estimated/approximated value-function

reward function plotten und mit echten daten vergleichen
exploration strategy plotten und mit echten daten vergleichen

Violin plot: A violin plot showing the distribution of rewards across episodes or runs.
Box plot: A box plot showing the quantiles of rewards across episodes or runs.
Histogram: A histogram of the distribution of rewards or actions taken by the agent.

Stacked area plot: A stacked area plot showing the cumulative reward for different actions or states over episodes.
Treemap: A treemap showing the distribution of rewards across different states or actions.

3D scatter plot: A 3D scatter plot of the Q-values or state-action values in the state-action-reward space.

in welchen state wird am meisten exploriert -> bestimmen anhand der samples
welche action wird am meisten exploriert -> bestimmen anhand der samples

plot frequency/distribution of decisions (make use of bins) -> trajectory length
dynamische bin size verwenden, anhand von relativen abweichung anstatt absoluten größen
clustering von werten, wenn bspw. 2-dim (scatter)

highlight min, max within all plots and mark mean as well as median if meaningful
alternative plot for trajectory (maybe number of errors etc.)
alternative plot for rewards (maybe max positive and max negative etc.)
generate another reward plot with the avg. over 100 (n) Iterations / acc. over 100 (n) Iterations
Reward normieren über Länge der Trajektorie
reward kurve, wird keine konvergenz gegen einen konstanten wert besitzen, aber gegen einen reward
akkumulierte kurve stabiler, kann nur steigen
irgendwie differenz bilden, sodass die kurve, sofern sie gegen eine steigung konvergiert auch gegen eine bound konvergiert?!
'''
