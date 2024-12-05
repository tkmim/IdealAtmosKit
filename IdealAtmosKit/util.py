import time
import logging


# set up the logger and set the logging level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Timer(object):
    """A function to measure the time of a block of code

    Parameters
    ----------
    name : str
        Name of a block to be measured.
    verbose : bool, optional
        If True, print the elapsed time. If False, log the elapsed time. Default is True.

    ===== Example =====

    with Timer('name of the block'):
        # code to measure

    """

    def __init__(self, name: str, verbose=True):
        self.verbose = verbose
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            if self.msecs > 1000 * 60:
                print(self.name, ": elapsed time: %.3f min" % (self.msecs / 60000.0))
            elif self.msecs > 1000:
                print(self.name, ": elapsed time: %.3f s" % (self.msecs / 1000.0))
            else:
                print(self.name, ": elapsed time: %.3f ms" % self.msecs)
        else:
            # if verbose is False, log the elapsed time
            if self.msecs > 1000 * 60:
                logger.info(self.name + ": elapsed time: %.3f min" % self.msecs / 60000.0)
            elif self.msecs > 1000:
                logger.info(self.name + ": elapsed time: %.3f s" % self.msecs / 1000.0)
            else:
                logger.info(self.name + ": elapsed time: %.3f ms" % self.msecs)
