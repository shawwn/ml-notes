# https://www.tensorflow.org/guide/data_performance#reproducing_the_figures
"""
This dataset provides samples of shape [[2, 1], [2, 2], [2, 3]] and of type [tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32]. Each sample is:


(
  [("Open"), ("Read")],
  [(t0, d), (t0, d)],
  [(i, e, -1), (i, e, s)]
)
Where:

Open and Read are steps identifiers
t0 is the timestamp when the corresponding step started
d is the time spent in the corresponding step
i is the instance index
e is the epoch index (number of times the dataset has been iterated)
s is the sample index
"""

import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


def now():
  return time.perf_counter()

def wait(secs):
  t0 = now()
  time.sleep(secs)
  return t0, secs

class OrderedDefaultDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        #in python3 you can omit the args to super
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory
    def __missing__(self, key):
        self[key] = value = self.default_factory()
        return value


class TimelineStep:
  def __init__(self):
    self.times = []
    self.values = []

  def add(self, time_start, time_spent, instance_index=0, epoch_index=0, sample_index=-1):
    self.times += [(time_start, time_spent)]
    self.values += [(instance_index, epoch_index, sample_index)]


class Timeline:
  def __init__(self):
    self.steps = OrderedDefaultDict(default_factory=lambda: TimelineStep())
  def add(self, step_name, time_start, time_spent, instance_index=0, epoch_index=0, sample_index=-1):
    self.steps[step_name].add(time_start=time_start, time_spent=time_spent, instance_index=instance_index, epoch_index=epoch_index, sample_index=sample_index)
  def get_timeline(self):
    steps = []
    times = []
    values = []
    for step, ts in self.steps.items():
      step = step.encode('utf8')
      for t, v in zip(ts.times, ts.values):
        steps += [tuple([step])]
        times += [t]
        values += [v]
    return {'steps': steps, 'times': times, 'values': values}


def test_timeline():
  tim = Timeline()
  t0 = now()
  for i in range(10):
    t = t0 + 2*i
    for phase in 'Open Read Map Train'.split():
      d = np.random.uniform()
      tim.add(phase, t, d, i); t += d
  return tim

def make_test_timeline():
  i = 0
  e = 0
  s = 0
  # tl = (
  #   [("Open"), ("Read")],
  #   [(t0, d), (t0, d)],
  #   [(i, e, -1), (i, e, s)]
  # )
  t0 = now()
  time.sleep(0.3)
  d = now() - t0
  steps += [("Open")]
  times += [(t0, d)]
  values += [(i, e, -1)]

  time.sleep(0.1)
  t0 = now()
  time.sleep(0.3)
  d = now() - t0
  steps += [("Read")]
  times += [(t0, d)]
  values += [(i, e, s)]
  return {'steps': steps, 'times': times, 'values': values}


def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # convert to numpy
    timeline['steps'] = np.array(timeline['steps'], dtype=np.bytes_)
    timeline['times'] = np.array(timeline['times'], dtype=np.float32)
    timeline['values'] = np.array(timeline['values'], dtype=np.int32)
    # Remove invalid entries (negative times, or empty steps) from the timelines
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]
    steps = timeline['steps'][invalid_mask]
    times = timeline['times'][invalid_mask]
    values = timeline['values'][invalid_mask]

    # Get a set of different steps, ordered by the first time they are encountered
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # Shift the starting time to 0 and compute the maximal time value
    min_time = times[:,0].min()
    times[:,0] = (times[:,0] - min_time)
    end = max(width, (times[:,0]+times[:,1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(17.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # Get timings and annotation for the given step
        entries_mask = np.squeeze(steps==step)
        serie = np.unique(times[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")

if __name__ == '__main__':
  tim = test_timeline()
  draw_timeline(tim.get_timeline(), 'test');
  plt.show()
