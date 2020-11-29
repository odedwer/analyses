from abc import ABC, abstractmethod


class BaseETParser(ABC):
    """
    Base class for ET parsers, cannot be instantiated (abstract). To use, inherit from this class and override methods
    """

    # there might be a need to define more "must have" properties such as this
    blink = False

    @property
    def TIME(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_binocular(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_empty_sample(cls, time):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_sample(cls, line):
        """
        parses a sample from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_msg(cls, line):
        """
        parses a message line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_input(cls, line):
        """
        parses a trigger line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_fixation(cls, line):
        """
        parses a fixation line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_saccade(cls, line):
        """
        parses a saccade line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_blinks(cls, line):
        """
        parses a blink line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def parse_recordings(cls, line):
        """
        parses a recording start/end line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @classmethod
    @abstractmethod
    def is_sample(cls, line):
        """
        checks if a line is a sample line
        :param line: line to check
        :return: True if line is sample, else False
        """
        pass
