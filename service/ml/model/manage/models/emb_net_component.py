from .base_net_component import BaseNetComponent


class EmbNetComponent(BaseNetComponent):
    def __init__(self, embeddings_path: str, embeddings_size: int):
        super().__init__()