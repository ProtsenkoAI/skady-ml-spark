def add_parent_to_path():
    from sys import path as sys_path
    from os import path as os_path
    curr_dir = sys_path[0]
    parent_dir = os_path.dirname(curr_dir)
    sys_path.insert(0, parent_dir)


add_parent_to_path()
