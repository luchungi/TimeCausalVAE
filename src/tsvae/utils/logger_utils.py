import logging


def get_console_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)
    return logger
