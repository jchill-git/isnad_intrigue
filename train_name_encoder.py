from typing import Optional

import json
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from transformers import logging

from helpers.data import read_isnad_names, read_isnad_data, split_data
from helpers.utils import invert_list

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.set_verbosity_error()

class NamesDataset(Dataset):
    def __init__(
        self,
        isnad_names_path: str,
        isnad_labels_path: str,
        test_mentions_path: Optional[str],
        is_train: bool = True
    ):
        # load data
        isnad_names = read_isnad_names(isnad_names_path)
        isnad_mention_ids, disambiguated_ids, _ = read_isnad_data(
            isnad_names_path,
            isnad_labels_path
        )

        # split data
        with open(test_mentions_path, "r") as test_mentions_path:
            test_mentions = json.load(test_mentions_path)
        mentions_split = invert_list(test_mentions) if is_train else test_mentions
        isnad_mention_ids, disambiguated_ids = split_data(
            isnad_mention_ids,
            disambiguated_ids,
            mentions_split
        )

        # only include disambiguated samples
        isnad_names_flattened = sum(isnad_names, [])
        mention_ids_flattened = sum(isnad_mention_ids, [])
        names = []
        labels = []
        for index, (name, mention_id) in enumerate(zip(
            isnad_names_flattened, mention_ids_flattened
        )):
            if mention_id in disambiguated_ids:
                names.append(name)
                labels.append(mention_id)

        self.names = names
        self.labels = labels
        assert len(self.names) == len(self.labels)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index: int):
        return self.names[index], self.labels[index]


def test_model(test_dataloader, tokenizer, model, criterion, max_samples=5):
    model.eval()

    # load data
    test_names, test_labels = next(iter(test_dataloader))
    test_names = test_names[:max_samples]
    test_labels = test_labels[:max_samples]

    # embed names
    test_tokens = tokenizer(
        test_names,
        is_split_into_words=False,
        max_length=wandb.config["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        test_outputs = model(**test_tokens).last_hidden_state

    # pool outputs
    if wandb.config["pooling_method"] == "cls":
        test_outputs = test_outputs[:, 0, :]
    if wandb.config["pooling_method"] == "mean":
        test_outputs = torch.mean(test_outputs, dim=1)

    # calculate loss
    loss = torch.tensor(0, dtype=torch.float)
    for test_key_output, test_key_label in zip(test_outputs, test_labels):
        test_key_output = test_key_output.unsqueeze(0)
        target = get_target(test_key_label, test_labels)
        loss += criterion(test_key_output, test_outputs, target)

    return loss.item() / len(test_labels)


def get_target(query_label, key_labels):
    return torch.tensor([
        1 if key_label == query_label else -1
        for key_label in key_labels
    ], device=DEVICE)


if __name__ == "__main__":
    wandb.init(
        project="isnad_contrastive_learning",
        entity="kylesayrs",
        config={
            "pretrained_model": "lanwuwei/GigaBERT-v3-Arabic-and-English",
            "num_epochs": 5,
            "batch_size": 32,
            "max_length": 32,
            "learning_rate": 5e-7,
            "pooling_method": "cls",
            "cosine_margin": 0.0,
            "batch_logging_rate": 5,
            "save_path": "nem_{epoch:.3f}_{loss:.3f}.pth"
        },
        mode="online"
    )
    assert wandb.config["pooling_method"] in ["cls", "mean"]

    # create dataset and data loaders
    train_dataset = NamesDataset(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "test_mentions.json",
        is_train=True
    )

    test_dataset = NamesDataset(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "test_mentions.json",
        is_train=False
    )

    query_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda data: [
            data[0][0],
            data[0][1],
        ]
    )

    key_dataloader = DataLoader(
        train_dataset,
        batch_size=wandb.config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda data: [
            [name for name, _ in data],
            [label for _, label in data],
        ]
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
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
    criterion = torch.nn.CosineEmbeddingLoss(margin=wandb.config["cosine_margin"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

    for epoch_num in range(wandb.config["num_epochs"]):
        train_losses = []
        test_losses = []

        for batch_num, ((query_name, query_label), (key_names, key_labels)) in enumerate(zip(
            query_dataloader, key_dataloader
        )):
            # reset gradient
            model.train()
            optimizer.zero_grad()

            # embed query
            query_tokens = tokenizer(
                query_name,
                max_length=wandb.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            query_output = model(**query_tokens).last_hidden_state

            # embed keys
            key_tokens = tokenizer(
                key_names,
                is_split_into_words=False,
                max_length=wandb.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            key_outputs = model(**key_tokens).last_hidden_state

            # pool outputs
            if wandb.config["pooling_method"] == "cls":
                query_output = query_output[:, 0, :]
                key_outputs = key_outputs[:, 0, :]
            if wandb.config["pooling_method"] == "mean":
                query_output = torch.mean(query_output, dim=1)
                key_outputs = torch.mean(key_outputs, dim=1)

            # compute loss and backpropagate
            target = get_target(query_label, key_labels)
            loss = criterion(query_output, key_outputs, target)
            loss.backward()
            optimizer.step()

            # log
            if batch_num % wandb.config["batch_logging_rate"] == 0:
                test_loss = test_model(test_dataloader, tokenizer, model, criterion)

                train_losses.append(loss.item())
                test_losses.append(test_loss)
                wandb.log({"train_loss": loss.item(), "test_loss": test_loss})
                print(
                    f"[{epoch_num}, {batch_num}]: "
                    f"(train_loss): {loss.item():.3f} "
                    f"(+/- mix): {sum(target)/len(target):.3f} "
                    f"(test_loss): {test_loss:.3f}"
                )

        #average_loss = sum(losses) / len(losses)
        #save_path = wandb.config["save_path"].format(epoch=epoch, loss=average_loss)
        #torch.save(model.state_dict(), save_path)
