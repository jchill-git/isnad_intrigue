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

    query_dataloader = DataLoader(
        names_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    key_dataloader = DataLoader(
        names_dataset,
        batch_size=wandb.config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # set up model and loss
    tokenizer = BertTokenizerFast.from_pretrained(
        "lanwuwei/GigaBERT-v3-Arabic-and-English",
        do_lower_case=True,
        to=DEVICE,
    )
    model = BertModel.from_pretrained("lanwuwei/GigaBERT-v3-Arabic-and-English").to(DEVICE)
    criterion = torch.nn.TripletMarginLoss(margin=1)

    for epoch in range(wandb.config["num_epochs"]):
        for (query_name, query_label), (key_names, key_labels) in zip(query_dataloader, key_dataloader):

            # collate
            query_name = query_name[0]
            key_names = list(key_names)

            print(query_name)
            print(query_label)
            print(key_names)
            print(key_labels)

            query_tokens = tokenizer(query_name, return_tensors="pt")
            print(query_tokens)
            query_output = model(**query_tokens)[0]
            query_output = query_output[0] # take cls token
            print(query_output)

            key_tokens = tokenizer(key_names, return_tensors="pt")
            print(key_tokens)
            key_outputs = model(**key_tokens)[0]
            key_outputs = key_outputs[0] # take cls token
            print(key_outputs)

            positive_outputs = key_outputs[query_label == key_labels]
            negative_outputs = key_outputs[query_label != key_labels]

            loss = criterion(query_output, positive_outputs, negative_outputs)

            break
        break
