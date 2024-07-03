import loguru
from langchain.callbacks import FileCallbackHandler

# Create a logger instance
logger = loguru.logger
logger.remove()
logfile = "/logfile.log"

# Configure the logger
logger.add(
    logfile,
    level="DEBUG",
    colorize=False,
    enqueue=True,
)

handler = FileCallbackHandler(logfile)

# # # Log messages using the logger variable
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
