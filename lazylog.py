import logging.config
import settings

from json_logger import JsonLogger

logging.setLoggerClass(JsonLogger)
logging.config.dictConfig(settings.LOGGING_CONFIG)


class LazyLog:
    def __init__(self):
        self._logger = logging.getLogger('default')

    @property
    def logger(self):
        return self._logger
