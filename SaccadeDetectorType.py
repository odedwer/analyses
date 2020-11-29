from enum import Enum

from SaccadeDetectors import *


class SaccadeDetectorType(Enum):
    """
    enum class for saccade detectiors. Values should be relevant class names.
    """
    ENGBERT_AND_MERGENTHALER = EngbertAndMergenthalerMicrosaccadeDetector
