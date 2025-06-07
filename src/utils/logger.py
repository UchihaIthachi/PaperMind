import logging
import sys
import os

# Add src directory to Python path to allow importing from src if this file is run directly for testing
# This assumes logger.py is in src/utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from src.config.app_config import LOG_LEVEL
except ImportError:
    # Fallback if running logger.py directly or src path not correctly set for some reason
    # This allows the module to be imported without error, but config won't be applied from app_config
    print("WARN: Could not import LOG_LEVEL from src.config.app_config. Defaulting LOG_LEVEL to INFO for logger setup.")
    LOG_LEVEL = "INFO"


# Store the formatter string as a module constant for potential reuse if needed
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"

_logging_configured = False

def initial_app_logging_config(level: str | None = None, force_reconfigure: bool = False):
    """
    Configures basic logging for the application using logging.basicConfig.
    This should be called once at the very start of the application.

    Args:
        level (str, optional): The logging level (e.g., "DEBUG", "INFO").
                               Defaults to LOG_LEVEL from app_config.
        force_reconfigure (bool): If True, will attempt to remove existing handlers
                                  and reconfigure. Useful for some environments like Streamlit
                                  that might pre-configure logging.
    """
    global _logging_configured
    if _logging_configured and not force_reconfigure:
        # logging.debug("Logging already configured.") # Use logging.debug if you want to see this
        return

    log_level_to_use = level if level else LOG_LEVEL
    log_level_enum = getattr(logging, log_level_to_use.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    # For environments like Streamlit that might have pre-configured root handlers,
    # removing them before basicConfig can sometimes help ensure your format takes precedence.
    if force_reconfigure:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close() # Close the handler

    logging.basicConfig(
        level=log_level_enum,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers # Pass handlers list (for Python 3.8+)
        # For older Python (<3.8), use stream=sys.stdout and then add other handlers manually if needed.
    )

    # Example: Quieten excessively noisy libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("httpcore").setLevel(logging.WARNING)
    # logging.getLogger("watchfiles").setLevel(logging.WARNING)

    _logging_configured = True
    logging.info(f"Root logging configured with level {log_level_to_use.upper()} and format '{DEFAULT_LOG_FORMAT}'.")


if __name__ == '__main__':
    print("Testing logger.py...")
    # Configure with DEBUG for this test
    initial_app_logging_config(level="DEBUG", force_reconfigure=True)

    # Get a logger for the current module (__name__ will be '__main__')
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message from logger.py.")
    logger.info("This is an info message from logger.py.")
    logger.warning("This is a warning message from logger.py.")

    # Test another module's logger (it should inherit the root config)
    another_module_logger = logging.getLogger("another.module")
    another_module_logger.info("Info message from another.module logger.")
    another_module_logger.debug("Debug message from another.module (should appear).")

    print("Logger test finished. Check console output.")
