from ...expose import RecsysTorchModel
from .mf_net_component import MFNetComponent
from .emb_net_component import EmbNetComponent


class MFWithUserEmbeddings(RecsysTorchModel):
    # TODO: investigate best network architecture to deal with embeddings
    def __init__(self, nusers, nitems, mf_hidden_size, embeddings_path: str, embeddings_size: int):
        super().__init__()
        self.mf_comp = MFNetComponent(nusers, nitems, mf_hidden_size)
        self.embed_comp = EmbNetComponent(embeddings_path, embeddings_size)






