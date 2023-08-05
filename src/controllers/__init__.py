REGISTRY = {}

from .basic_controller import BasicMAC
from .i_controller import IMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["i_mac"] = IMAC