# python -m pysc2.bin.agent --map <Map> --agent <Agent>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

class OurAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(OurAgent, self).step(obs)
    #----------------------------------#
    # obs contains state, feature maps, rewards, available_actions

    RL Algorithm Here

    #----------------------------------#
    return action





# Play Beacon mini game.
# python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.scripted_agent.MoveToBeacon

# Play replay
# python -m pysc2.bin.play --replay <path-to-replay>
#
# e.g.
# python -m pysc2.bin.play --replay /Applications/StarCraft\ II/Replays/MoveToBeacon/MoveToBeacon_2017-11-01-01-34-11.SC2Replay
#
# random.
# python -m pysc2.bin.play --replay /Applications/StarCraft\ II/Replays/RandomAgent/Simple64_2017-10-23-08-10-40.SC2Replay

# Manual play
# python -m pysc2.bin.play --map MoveToBeacon