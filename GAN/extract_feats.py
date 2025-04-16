import os
from tqdm import tqdm
import random
import re
import torch
from transformers import BertTokenizer, BertModel

random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_numbers_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def encode_prompt(prompt):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    # Forward pass through BERT
    outputs = model(**inputs)

    # Extract the CLS token embedding as a fixed-length representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    # to tensor
    cls_embedding = cls_embedding.detach()
    return cls_embedding

def combine_feats(prompt, file=None):
    # Encode the prompt
    prompt_embedding = encode_prompt(prompt)

    # Extract features from the file
    if file is not None:
        stats = extract_numbers_feats(file)
    else:
        stats = extract_numbers(prompt)
    # Convert the features to a tensor
    stats = torch.tensor(stats)
    print(stats.shape)
    print(prompt_embedding.shape)
    # Combine the prompt embedding and the features
    combined_feats = torch.cat((prompt_embedding, stats))
    return combined_feats

def extract_feats(desc, feature_extractor, file=None):
    if feature_extractor == "extract_numbers":
        feats_stats = extract_numbers(desc)
    elif feature_extractor == "encode_prompt":
        feats_stats = encode_prompt(desc)
    elif feature_extractor == "combine_feats":
        feats_stats = combine_feats(desc, file)
    else: # return error
        print(feature_extractor, "Invalid feature extractor, back to default")
        feats_stats = extract_numbers(desc)

    return feats_stats
