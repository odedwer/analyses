# In order to add parsers, implement a new class like MonocularNoVelocityParser
from BaseETParser import BaseETParser
from EyeEnum import Eye


class BinocularNoVelocityParser(BaseETParser):
    """
    parser for EyeLink 1000 Monocular recording with no velocity data
    """

    # constants for column names, allow for quick and easy changes
    TYPE = "type"
    PEAK_VELOCITY = "peak velocity"
    AMPLITUDE = "amplitude"
    END_Y = "end y"
    END_X = "end x"
    START_Y = "start y"
    START_X = "start x"
    AVG_PUPIL_SIZE = "avg pupil size"
    AVG_Y = "avg y"
    AVG_X = "avg x"
    DURATION = "duration"
    END_TIME = "end time"
    START_TIME = "start time"
    EYE_STR = "eye"
    TRIGGER = "trigger"
    MSG = "message"
    RIGHT_PUPIL_SIZE = "right pupil size"
    RIGHT_Y = "right y"
    RIGHT_X = "right x"
    LEFT_PUPIL_SIZE = "left pupil size"
    LEFT_Y = "left y"
    LEFT_X = "left x"
    TIME = "time"
    MISSING_VALUE = -1

    @property
    def is_binocular(self):
        return True

    @classmethod
    def parse_sample(cls, line):
        """
        parses a sample line from the EDF
        """
        return {cls.TIME: int(line[0]), cls.LEFT_X: float(line[1]), cls.LEFT_Y: float(line[2]),
                cls.LEFT_PUPIL_SIZE: float(line[3]), cls.RIGHT_X: float(line[4]), cls.RIGHT_Y: float(line[5]),
                cls.RIGHT_PUPIL_SIZE: float(line[6])}

    @classmethod
    def parse_msg(cls, line):
        """
        parses a message line from the EDF
        """
        return {cls.TIME: int(line[1]), cls.MSG: "".join(line[2:-1])}

    @classmethod
    def parse_input(cls, line):
        """
        parses a trigger line from the EDF
        """
        return {cls.TIME: int(line[1]), cls.TRIGGER: int(line[2])}

    @classmethod
    def parse_fixation(cls, line):
        """
        parses a fixation line from the EDF
        """
        return {cls.EYE_STR: line[1], cls.START_TIME: int(line[2]), cls.END_TIME: int(line[3]),
                cls.DURATION: int(line[4]), cls.AVG_X: float(line[5]), cls.AVG_Y: float(line[6]),
                cls.AVG_PUPIL_SIZE: float(line[6])}

    def get_empty_sample(cls, time):
        return {cls.TIME: time, cls.LEFT_X: cls.MISSING_VALUE, cls.LEFT_Y: cls.MISSING_VALUE,
                cls.LEFT_PUPIL_SIZE: cls.MISSING_VALUE, cls.RIGHT_X: cls.MISSING_VALUE, cls.RIGHT_Y: cls.MISSING_VALUE,
                cls.RIGHT_PUPIL_SIZE: cls.MISSING_VALUE}

    @classmethod
    def parse_saccade(cls, line):
        """
        parses a saccade line from the EDF
        """
        return {cls.EYE_STR: line[1], cls.START_TIME: int(line[2]), cls.END_TIME: int(line[3]),
                cls.DURATION: int(line[4]), cls.START_X: float(line[5]), cls.START_Y: float(line[6]),
                cls.END_X: float(line[6]), cls.END_Y: float(line[7]), cls.AMPLITUDE: float(line[8]),
                cls.PEAK_VELOCITY: float(line[9])}

    @classmethod
    def parse_blinks(cls, line):
        """
        parses a blink line from the EDF
        """
        cls.blink = True
        return {cls.EYE_STR: line[1], cls.START_TIME: int(line[2]), cls.END_TIME: int(line[3]),
                cls.DURATION: int(line[4])}

    @classmethod
    def parse_recordings(cls, line):
        """
        parses a recording start/end line from the EDF
        """
        return {cls.TYPE: line[0], cls.TIME: int(line[1])}

    @classmethod
    def is_sample(cls, line):
        try:
            int(line[0])
            return True
        except:
            return line[-2] == '.....'

    @classmethod
    def get_type(cls):
        return Eye.BOTH

    # has to be last in order to find the parsing methods
    parse_line_by_token = {'INPUT': parse_input, 'MSG': parse_msg, 'ESACC': parse_saccade, 'EFIX': parse_fixation,
                           "EBLINK": parse_blinks}
