def add_parent_dir_to_path():
    import sys, os
    curr_dir = sys.path[0]
    parent_dir = os.path.dirname(curr_dir)
    sys.path.insert(0, parent_dir)

add_parent_dir_to_path()