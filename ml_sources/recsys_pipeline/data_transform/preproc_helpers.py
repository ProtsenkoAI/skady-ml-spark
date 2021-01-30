def wrap_in_list_if_number(func):
    def wrapper(self, data_arg, *args, **kwargs):
        if isinstance(data_arg, (int, float, complex)):
            data_arg = [data_arg]
        return func(self, data_arg, *args, **kwargs)
    return wrapper