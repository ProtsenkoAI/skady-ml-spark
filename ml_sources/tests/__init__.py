def update_path():
    from sys import path as sys_path
    from os import path as os_path
    curr_dir = sys_path[0]
    parent_dir = os_path.dirname(curr_dir)
    sys_path.insert(0, parent_dir)


update_path()
from . import test_data