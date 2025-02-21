import functools
import logging
import os
import time
import warnings

from astropy import log as astrolog

from .colors import colors


def handle_warnings(message, category, filename, lineno, file=None, line=None):
    """
    Convert warnings to log messages.

    Parameters
    ----------
    message : Warning
        The warning message
    category : Warning
        The warning category
    filename : str
        The file that raised the warning
    lineno : int
        The line number that raised the warning
    file : file object, optional
        The file object
    line : str, optional
        The line of code that raised the warning
    """
    warning_message = f"{category.__name__}: {str(message)} (in {filename}:{lineno})"
    astrolog.warning(warning_message)


def catch_warnings(func):
    """
    Decorator to catch warnings and redirect them to the logger.

    This will catch all warnings, including those from numpy, astropy,
    and other dependencies.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function to catch warnings"""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = handle_warnings
            logging.captureWarnings(True)
            return func(*args, **kwargs)

    return wrapper


class warning_to_log:
    """Context manager to temporarily redirect warnings to logger"""

    def __enter__(self):
        """Enter the context manager"""
        self.showwarning = warnings.showwarning
        warnings.simplefilter("always")
        warnings.showwarning = handle_warnings
        logging.captureWarnings(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager"""
        warnings.showwarning = self.showwarning
        logging.captureWarnings(False)


class Logger:
    """
    Logger class for logging messages to a file and the terminal.
    """

    # class variable to store the log file path
    log_file_path = None

    def __init__(self, logfile: str | None = None):
        """
        Initialize the logger.

        Parameters
        ----------
        logfile : str, optional
            Path to the log file. If None, only console logging is enabled.
        """
        self.messages = []
        self.colors = {
            "staging": colors.fg._lightblue_,
            "information": colors.fg._lightgrey_,
            "pass": colors.fg._lightgreen_,
            "warning": colors.fg._yellow_,
            "severe": colors.fg._lightred_,
        }

        if logfile is None:
            logfile = os.path.join(os.getcwd(), "scousepy.log")

        Logger.log_file_path = logfile

        # Ensure the directory exists
        os.makedirs(os.path.dirname(logfile), exist_ok=True)

        # Add file handler to astropy logger
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        astrolog.setLevel(logging.INFO)
        astrolog.handlers = []
        astrolog.addHandler(file_handler)

        # if the log file is empty, state that the logger has been initialized
        if os.path.getsize(logfile) == 0:
            self.printlog(
                message="logger initialized", status="staging", print_to_terminal=False
            )

    @classmethod
    def get_log_file_path(cls) -> str:
        """
        Get the log file path.

        Returns
        -------
        str
            The log file path
        """
        if cls.log_file_path is None:
            raise RuntimeError("log file path not set")
        return cls.log_file_path

    def printlog(
        self,
        message: str,
        status: str = "pass",
        timetaken: float | None = None,
        print_to_terminal: bool = True,
    ):
        """
        Log a message with the given status level using astropy's logger.

        Parameters
        ----------
        message : str
            The message to log
        status : str, optional
            The message status ('staging', 'information','pass', 'warning', 'severe')

            'staging' : used to indicate the start of a new stage in the code
            'information' : used to indicate information about the code
            'pass' : used to indicate that a step has been completed successfully
            'warning' : used to indicate a warning
            'severe' : used to indicate a severe error

        timetaken : float, optional
            The time taken to complete the task. it is given in seconds.

        print_to_terminal : bool, optional
            Whether to print the message to the terminal
        """
        if status not in self.colors:
            status = "pass"

        # Store message in history
        self.messages.append(
            {
                "message": message,
                "status": status,
                "timestamp": time.time(),
                "timetaken": timetaken,
            }
        )

        # Format colored message for terminal
        if print_to_terminal:
            color = self.colors[status]
            time_str = None
            if timetaken is not None:
                if timetaken > 3600:
                    time_str = f"{timetaken / 3600:.2f} hours"
                elif timetaken > 60:
                    time_str = f"{timetaken / 60:.2f} minutes"
                else:
                    time_str = f"{timetaken:.2f} seconds"
                formatted_msg = f"{color}{message} ({time_str}){colors._endc_}"
            else:
                formatted_msg = f"{color}{message}{colors._endc_}"

            # Add dividers for staging messages
            if status == "staging":
                divider = "-" * len(message)
                formatted_msg = (
                    f"\n{color}{divider}\n{message}\n{divider}{colors._endc_}"
                )

            print(f"{formatted_msg}\n")

        # Log to astropy logger
        if status == "staging":
            astrolog.info(f"{status}: {message}")
        elif status == "information":
            astrolog.info(f"{status}: {message}")
        elif status == "pass":
            astrolog.info(f"{status}: {message}")
        elif status == "warning":
            astrolog.warning(f"{status}: {message}")
        else:
            astrolog.error(f"{status}: {message}")

    def close_logger(self):
        """
        Close the logger.
        """
        astrolog.handlers = []
        astrolog.setLevel(logging.INFO)
