import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m"
    }
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{msg}{self.RESET}"

class Logger:
    _logger = None
    _logger_level=logging.INFO
    @staticmethod
    def getRootPath():
        '''
         always find the path from log_utils.py
         __file__ :log_utils.py
        '''
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    @staticmethod
    def get_logger(log_file="app.log", backup_count=30, log_level=_logger_level):
        log_dir=Logger.getRootPath()
        log_dir = os.path.join(log_dir, "logs")
        if Logger._logger is None:
            Logger._logger = Logger._create_logger(log_file, log_dir, backup_count, log_level)
        return Logger._logger

    @staticmethod
    def _create_logger(log_file, log_dir, backup_count, log_level):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        logger = logging.getLogger("AppLogger")
        logger.setLevel(log_level)

        handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=backup_count)
        handler.suffix = "%Y-%m-%d"

        colorformatter = ColorFormatter(
            '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.handlers.clear()  # clear old handlers
        logger.propagate = False  # in case the root logger will output once again.

        #setup file handler
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        #setup concole handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(colorformatter)
        logger.addHandler(console_handler)
        return logger
