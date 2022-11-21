import wandb

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel

from helpers.data import read_isnad_names, read_isnad_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NamesDataset(Dataset):
    def __init__(self, isnad_names_path: str, isnad_labels_path: str):
        isnad_names = read_isnad_names(isnad_names_path)
        isnad_mention_ids, _, _ = read_isnad_data(
            isnad_names_path,
            isnad_labels_path
        )

        self.isnad_names_flattened = sum(isnad_names, [])
        self.mention_ids_flattened = sum(isnad_mention_ids, [])
        assert len(self.isnad_names_flattened) == len(self.mention_ids_flattened)


    def __len__(self):
        return len(self.isnad_names_flattened)

    def __getitem__(self, index: int):
        name = self.isnad_names_flattened[index]
        label = self.mention_ids_flattened[index]

        return name, label

class ContrastiveLoss(nn.Module):
    def __init__(self):
        self.similarity = torch.nn.CosineSimilarity(dim=2)

if __name__ == "__main__":
    wandb.init(
        project="isnad_contrastive_learning",
        entity="kylesayrs",
        config={
            "num_epochs": 3,
            "batch_size": 32
        },
        mode="disabled"
    )

    # create dataset and data loaders
    names_dataset = NamesDataset(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json"
    )

    dataloader = DataLoader(
        names_dataset,
        batch_size=wandb.config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    """
    # set up model and loss
    tokenizer = BertTokenizerFast.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English", do_lower_case=True)
    model = BertModel.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English").to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    for epoch in wandb.config["num_epochs"]:
        for names, labels in dataloader:
            names = names.to(DEVICE)
            labels = labels.to(DEVICE)

            tokens = tokenizer(names)
            outputs = model(**tokens)
            cls_output = outputs[0]
    """
