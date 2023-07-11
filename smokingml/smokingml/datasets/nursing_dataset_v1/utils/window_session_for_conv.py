from . import window_session
from ..import WINSIZE

def window_session_for_conv(session_id):
    return window_session(session_id).reshape(-1, 3, WINSIZE)