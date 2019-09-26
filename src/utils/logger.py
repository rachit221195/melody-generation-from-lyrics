import logging

from datetime import datetime
from pythonjsonlogger import jsonlogger

DEFAULT_LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGGING_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(
            log_record, record, message_dict
        )
        if not log_record.get("timestamp"):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_record["timestamp"] = now
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname


class Logger:
    name = ""
    logger = None
    handlers = []

    def __init__(self, level, name, logger_format=DEFAULT_LOGGER_FORMAT,
                 json_formatter=True, filename=None):
        if Logger.logger is not None:
            return

        Logger.name = name

        Logger.logger = logging.getLogger(name)
        Logger.logger.setLevel(LOGGING_LEVELS.get(level, logging.INFO))

        if json_formatter:
            logger_format = CustomJsonFormatter(
                "(timestamp) (level) (name) (message)"
            )
        elif isinstance(logger_format, str):
            logger_format = logging.Formatter(logger_format)

        if filename is not None:
            Logger.handlers.append(file_handler(filename, level, logger_format))
        else:
            Logger.handlers.append(stream_handler(level, logger_format))
        Logger.logger.addHandler(Logger.handlers[0])

    def __str__(self):
        return repr(self) + repr(self.name)


def file_handler(filename, level, formatter):
    handler = logging.FileHandler(filename)

    handler.setLevel(LOGGING_LEVELS.get(level, logging.INFO))
    handler.setFormatter(formatter)

    return handler


def stream_handler(level, formatter):
    handler = logging.StreamHandler()

    handler.setLevel(LOGGING_LEVELS.get(level, logging.INFO))
    handler.setFormatter(formatter)

    return handler


def debug(message, extra={}):
    Logger.logger.debug(message, extra=extra)


def info(message, extra={}):
    Logger.logger.info(message, extra=extra)


def warn(message, extra={}):
    Logger.logger.warn(message, extra=extra)


def error(message, extra={}):
    Logger.logger.error(message, extra=extra)


def critical(message, extra={}):
    Logger.logger.critical(message, extra=extra)


def clearLogger():
    for handler in Logger.handlers:
        Logger.logger.removeHandler(handler)

    Logger.logger = None
    Logger.name = ""