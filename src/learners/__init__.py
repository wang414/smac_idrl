from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .z_learner import ZLearner
from .ma2ql_learner import Ma2qlLearner
from .i_learner import ILearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["z_learner"] = ZLearner
REGISTRY["ma2ql_learner"] = Ma2qlLearner
REGISTRY["i_learner"] = ILearner
