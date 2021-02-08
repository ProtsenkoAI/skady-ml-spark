from torch.utils import data as torch_data


class InteractDataset(torch_data.Dataset):
    def __init__(self, interacts, has_label=True):
        self.interacts = interacts
        self.user_colname = "user_id"
        self.item_colname = "anime_id"
        self.label_colname = "rating"
        self.has_label = has_label

    def __len__(self):
        return len(self.interacts)

    def __getitem__(self, idx):
        row = self.interacts.iloc[idx]
        user_and_item = row[[self.user_colname, self.item_colname]].values
        if self.has_label:
            label = row[self.label_colname]
            return user_and_item, label
        return user_and_item
