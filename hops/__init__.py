
__version__ = '4.0.0'
__message__ = ('Version 4.0 is now online with new features!'
               '\n- Support for a local GAIA database.'
               '\n- Option to stack the reduced images.'
               )

from .__run__ import run_app


def __get_abspath__():
    import os
    return os.path.abspath(os.path.dirname(__file__))
