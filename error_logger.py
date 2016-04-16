import inspect


ERROR_TABLE = {
  'unknown': -1,
  'assertion': 1
}


def _get_stack_string():
  """Returns a string of the caller's grandparent on the stack.

  The string will contain the file name, line number, and function two levels
  up the calling stack. Call this by an assertion or error function to get info
  about the function that called that assertion or error.

  Returns:
    The stack record string.
  """
  callerframerecord = inspect.stack()[2]
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  return 'in file {} on line {} ({})'.format(
      info.filename, info.lineno, info.function)


def assert_or_die(condition, message=None):
  """If the condition is false, the program will exit.

  Args:
    condition: a value that evaluates to True or False (or None) by python type
        standards.
    message: (optional) the message that will be logged if the condition is
        false.
  """
  if not condition:
    log_error('Assertion error {}.'.format(_get_stack_string()))
    if message:
      log_error(message)
    exit(ERROR_TABLE['assertion'])


def error_and_die(message=None):
  """Logs the message (if provided) and exists with an error.
  
  Args:
    message: (optional) the message that will be logged.
  """
  log_error('Error {}.'.format(_get_stack_string()))
  if message:
    log_error(message)
  exit(ERROR_TABLE['unknown'])


# TODO: These log functions should either print or save to a log file.
def log_message(message):
  """Prints the given message."""
  print message


def log_warning(warning):
  """Prints the given warning message."""
  print warning


def log_error(error):
  """Prints the given error message."""
  print error
