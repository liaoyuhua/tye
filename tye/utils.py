"""
Utility functions for tye.
"""


def tolist(x):
    if isinstance(x, list):
        return x
    else:
        return [x]
