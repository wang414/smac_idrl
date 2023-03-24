REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
from .i_rnn_agent import IRNNAgent
REGISTRY["irnn"] = IRNNAgent
from .i_rnn_zagent import IRNNZAgent
REGISTRY["izrnn"] = IRNNZAgent