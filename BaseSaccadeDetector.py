from abc import ABC, abstractmethod


class BaseSaccadeDetector(ABC):
    @classmethod
    @abstractmethod
    def detect_saccades(cls, saccade_data, sf):
        """
        detects saccades/microsaccads in the given data
        :param saccade_data: time X Position (X,Y) matrix
        :param sf: sampling frequency
        :return: dataframe of detected saccades
        """
        pass
