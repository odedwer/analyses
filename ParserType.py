from enum import Enum

from EyeTrackingParsers import *


class ParserType(Enum):
    """
    Enum class for ET parsers. Values should be class names.
    """
    MONOCULAR_NO_VELOCITY = BinocularNoVelocityParser
