"""
usage: python3 embed_names.py nameData/names.csv checkpoints/nem_0_0.000 name_embeddings.json
"""

import json
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, logging

from helpers.data import read_isnad_names

parser = argparse.ArgumentParser()
parser.add_argument("isnad_names_path", type=str)
parser.add_argument("model_dir_path", type=str)
parser.add_argument("out_file_path", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--tokenizer", type=str, default="lanwuwei/GigaBERT-v3-Arabic-and-English")
parser.add_argument("--max_length", type=int, default=32)
parser.add_argument("--pooling_method", type=str, default="mean")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.set_verbosity_error()

def embed_names(dataloader, tokens, model):
    outputs = []
    for names_batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        tokens = tokenizer(
            names_batch,
            is_split_into_words=False,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            names_outputs = model(**tokens).last_hidden_state
            outputs.append(names_outputs)

    return torch.cat(outputs, dim=0)

def write_outputs(outputs, isnad_names, out_file_path):
    writing_progress = tqdm.tqdm(total=len(outputs))
    with open(args.out_file_path, "w") as out_file:
        output_index = 0
        for isnad_index in range(len(isnad_names)):
            for mention_index in range(len(isnad_names[isnad_index])):
                id = f"JK_000916_{isnad_index}_{mention_index}"
                embedding = outputs[output_index].tolist()

                data = {"id": id, "embedding": embedding}
                out_file.write(json.dumps(data) + "\n")

                output_index += 1
                writing_progress.update(1)

                if output_index >= len(outputs):
                    return

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.pooling_method in ["cls", "mean"]

    # load data
    isnad_names = read_isnad_names(args.isnad_names_path)
    names = sum(isnad_names.copy(), [])

    dataloader = DataLoader(
        names,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # set up model
    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer,
        do_lower_case=True,
        to=DEVICE,
    )
    model = BertModel.from_pretrained(args.model_dir_path).to(DEVICE)
    model.eval()

    # embed names
    outputs = embed_names(dataloader, tokenizer, model)

    # pooling
    if args.pooling_method == "cls":
        outputs = outputs[:, 0, :]
    if args.pooling_method == "mean":
        outputs = torch.mean(outputs, dim=1)

    # save outputs to file
    write_outputs(outputs, isnad_names, args.out_file_path)
