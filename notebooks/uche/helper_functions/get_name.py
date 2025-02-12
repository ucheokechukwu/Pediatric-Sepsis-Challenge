import re
import inspect
# import traceback


def get_name(a):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    return re.search(r'\w+\((\w+)\)', string).group(1)

# def get_name(var):
#     """will not work as an imported function"""
#     stack = traceback.extract_stack()
#     filename, lineno, function_name, code = stack[-2]
#     vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
#     return vars_name
