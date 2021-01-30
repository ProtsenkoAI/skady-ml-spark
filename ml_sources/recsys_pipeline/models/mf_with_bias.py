from torch import nn
import torch
from torch.nn import parameter

class MFWithBiasModel(nn.Module):
    def __init__(self, nusers, nitems, hidden_size=20):
        super().__init__()
        self.nusers = nusers
        self.nitems = nitems
        self.hidden_size = hidden_size

        self._init_layers(self.nusers, self.nitems, self.hidden_size)

    def _init_layers(self, nusers, nitems, hidden):
        self.user_factors = self._init_embedding(nusers, hidden)
        self.item_factors = self._init_embedding(nitems, hidden)
        self.user_biases = self._init_embedding(nusers, hidden)
        self.item_biases = self._init_embedding(nitems, hidden)

    def _init_embedding(self,  inp_size, out_size):
        return nn.Embedding(inp_size, out_size, sparse=True)

    def save_model(self, saver):
        model_kwargs = {"nusers": self.nusers, "nitems": self.nitems, "hidden_size": self.hidden_size}
        model_name = f"mf_with_bias_{nusers}_{nitems}_{hidden_size}"
        saver.save(model_name, self.state_dict(), model_kwargs)

    def add_users(self, nusers):
        self.nusers += nusers
        user_layers = [self.user_factors, self.user_biases]
        self._concat_new_weights_to_params(user_layers, nusers)
    
    def add_items(self, nitems):
        self.nitems += nitems
        item_layers = [self.item_factors, self.item_biases]
        self._concat_new_weights_to_params(item_layers, nitems)

    def _concat_new_weights_to_params(self, curr_layers, number_of_objects_added):
        for layer in curr_layers:
            new_weights = self._init_new_weights(number_of_objects_added)
            others_weights = layer.weight
            new_weight = torch.cat([others_weights, new_weights], dim=0)
            layer.weight = parameter.Parameter(new_weight)

    def _init_new_weights(self, number_of_objects_added):
        tensor = torch.zeros(number_of_objects_added, self.hidden_size)
        return nn.init.normal_(tensor)

    def forward(self, user, item):
        bias_sum = self.user_biases(user) + self.item_biases(item)
        user_emb = self.user_factors(user)
        item_emb = self.item_factors(item)

        product_with_bias = user_emb * item_emb + bias_sum
        return product_with_bias.sum(dim=1, keepdim=True)
        