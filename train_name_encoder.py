import wandb

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from transformers import logging

from helpers.data import read_isnad_names, read_isnad_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.set_verbosity_error()

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
            "pretrained_model": "lanwuwei/GigaBERT-v3-Arabic-and-English",
            "num_epochs": 3,
            "batch_size": 32,
            "max_length": 32,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "temperature": 1.0,
            "save_path": "name_embedding_model_{epoch}_{loss}.pth"
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
        collate_fn=lambda data: [
            data[0][0],
            data[0][1],
        ]
    )

    key_dataloader = DataLoader(
        names_dataset,
        batch_size=wandb.config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda data: [
            [name for name, _ in data],
            [label for _, label in data],
        ]
    )

    # set up model, optimizer, and loss
    tokenizer = BertTokenizerFast.from_pretrained(
        wandb.config["pretrained_model"],
        do_lower_case=True,
        to=DEVICE,
    )
    model = BertModel.from_pretrained(wandb.config["pretrained_model"]).to(DEVICE)
    criterion = torch.nn.CosineEmbeddingLoss().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=wandb.config["learning_rate"],
        momentum=wandb.config["momentum"]
    )

    for epoch in range(wandb.config["num_epochs"]):
        losses = []

        for (query_name, query_label), (key_names, key_labels) in zip(query_dataloader, key_dataloader):
            # reset gradient
            optimizer.zero_grad()

            # embed query
            query_tokens = tokenizer(
                query_name,
                max_length=wandb.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            query_output = model(**query_tokens).last_hidden_state
            query_output = query_output[:, 0, :] # take cls token

            # embed keys
            key_tokens = tokenizer(
                key_names,
                is_split_into_words=False,
                max_length=wandb.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            key_outputs = model(**key_tokens).last_hidden_state
            key_outputs = key_outputs[:, 0, :] # take cls token

            # compute loss and backpropagate
            target = torch.tensor([
                1 if key_label == query_label else -1
                for key_label in key_labels
            ])
            loss = criterion(query_output, key_outputs, target)
            loss.backward()
            optimizer.step()

            # log
            losses.append(loss.item())
            wandb.log({"loss": loss.item()})
            print(f"loss: {loss.item()}")

        average_loss = sum(losses) / len(losses)
        save_path = wandb.config["save_path"].format(epoch=epoch, loss=average_loss)
        torch.save(model.state_dict(), save_path)
