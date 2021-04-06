import dill
import os
import codecs


def dump_dill(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized = dill.dumps(obj)
    with open(path, "wb") as f:
        encoded = codecs.encode(serialized, "base64")
        f.write(encoded)


def load_dill(path):
    with codecs.open(path, "rb") as f:
        torch_obj_decoded = codecs.decode(f.read(), "base64")
    loaded_obj = dill.loads(torch_obj_decoded)
    return loaded_obj
