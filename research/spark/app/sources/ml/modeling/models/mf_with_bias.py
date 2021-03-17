import torch
from torch import nn


class MFWithBiasModel(nn.Module):
    def __init__(self, nusers, nitems, hidden_size):
        self.nusers = nusers
        self.nitems = nitems
        self.hidden_size = hidden_size
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.user_factors = self._init_embedding(self.nusers, self.hidden_size)
        self.item_factors = self._init_embedding(self.nitems, self.hidden_size)
        self.user_biases = self._init_embedding(self.nusers, self.hidden_size)
        self.item_biases = self._init_embedding(self.nitems, self.hidden_size)

    def _init_embedding(self, inp_size, out_size):
        return nn.Embedding(inp_size, out_size, sparse=True)

    def forward(self, users, items):
        bias_sum = self.user_biases(users) + self.item_biases(items)
        user_emb = self.user_factors(users)
        item_emb = self.item_factors(items)

        product_with_bias = user_emb * item_emb + bias_sum
        return product_with_bias.sum(dim=1, keepdim=True)

    # def add_users(self, nusers):
    #     self.nusers += nusers
    #     user_layers = [self.user_factors, self.user_biases]
    #     self._concat_new_weights_to_params(user_layers, nusers)
    #
    # def add_items(self, nitems):
    #     self.nitems += nitems
    #     item_layers = [self.item_factors, self.item_biases]
    #     self._concat_new_weights_to_params(item_layers, nitems)
    #
    # def get_init_kwargs(self):
    #     return {"nusers": self.nusers, "nitems": self.nitems, "hidden_size": self.hidden_size}
    #
    # def _concat_new_weights_to_params(self, curr_layers, number_of_objects_added):
    #     for layer in curr_layers:
    #         new_weights = self._init_new_weights(number_of_objects_added)
    #         others_weights = layer.weight
    #         new_weight = torch.cat([others_weights, new_weights], dim=0)
    #         layer.weight = nn.parameter.Parameter(new_weight)
    #
    # def _init_new_weights(self, number_of_objects_added):
    #     tensor = torch.zeros(number_of_objects_added, self.hidden_size)
    #     return nn.init.normal_(tensor)