import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import json
import os
import logging
import math


# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variable for transformers
os.environ['HF_HOME'] = 'C:\\Users\\brive_tut31mn\\.cache\\huggingface'

# Initialize tokenizer and model with debug logging
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.environ['HF_HOME'])
    bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.environ['HF_HOME'])
    logging.info("Tokenizer and BERT model loaded successfully.")
except Exception as e:
    logging.error("Failed to load tokenizer or BERT model: " + str(e))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_padding_mask(seq):
    # seq is typically [batch size, sequence length] with padded values as 0
    # Mask will have 1 where seq is 0 (padded), else 0
    return (seq == 0).to(torch.float32)

class MyDataset(Dataset):
    def __len__(self):
        return 100  # Number of items in the dataset

    def __getitem__(self, index):
        # Return a single item by index
        # This is a dummy implementation
        return {'input_ids': torch.tensor([1, 2, 3])}

# Define your dataset and DataLoader
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # Split the embedding into self.heads different pieces
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for all batches and heads at once
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


try:
    # Assuming you have defined `Transformer` and related setup correctly
    transformer = Transformer(...).to(DEVICE)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    logging.info("Starting training loop...")
    for epoch in range(5):  # Let's say we run for 5 epochs
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            logging.debug(f"Processing batch with {input_ids.size(0)} samples.")
            # continue your code...
except Exception as e:
    logging.error("An error occurred during training.", exc_info=True)

class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_mult, dropout, num_layers):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_mult, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_mult, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_mult * embed_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, mask):
        attention = self.attention(value, value, value, attn_mask=mask)[0]
        x = self.norm1(value + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out

# Example usage:
embed_size = 512
num_heads = 8
ff_hidden_mult = 4
dropout = 0.1
num_layers = 6
batch_first = True


# Transformer instantiation
transformer = Transformer(embed_size, num_heads, ff_hidden_mult, dropout, num_layers).to(DEVICE)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


model = Transformer(embed_size, num_heads, ff_hidden_mult, dropout, num_layers)
x = torch.randn(10, 32, embed_size)  # (sequence_length, batch_size, embed_size)
output = model(x, mask)
print(output.shape)  # Check the output shape

class CoQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            coqa_data = json.load(file)
            for entry in coqa_data['data']:
                story = entry['story']
                questions = entry['questions']
                answers = entry['answers']
                for question, answer in zip(questions, answers):
                    encodings = tokenizer.encode_plus(
                        question['input_text'], story,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    self.data.append({
                        'input_ids': encodings['input_ids'].squeeze(0).to(DEVICE),
                        'attention_mask': encodings['attention_mask'].squeeze(0).to(DEVICE),
                        'answers': answer['span_text']  # Assuming this is handled appropriately later
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Usage of the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_path = 'C:/Users/brive_tut31mn/.vscode/web projects/learning/GPT learning/qa_transformer/coqa-train-v1.0.json'
coqa_dataset = CoQADataset(dataset_path, tokenizer)
dataloader = DataLoader(coqa_dataset, batch_size=10, shuffle=True)

try:
    logging.info("Starting training loop...")
    for epoch in range(5):  # Number of epochs
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = create_attention_mask(input_ids)  # Generate mask for this batch

            outputs = transformer(input_ids, mask)  # Pass both input_ids and mask to your model

            # Assuming you have a way to get your labels ready for loss computation:
            labels = process_labels(batch['answers'])  # Define or update this function based on your needs
            labels = labels.to(DEVICE)
    
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            logging.debug(f"Processed batch with loss {loss.item()}.")

    def process_labels(labels):
        # Convert labels to tensor or any required format
        # This example assumes you are converting string labels to numerical indices
        return torch.tensor([label_to_index[label] for label in labels], device=DEVICE)


except Exception as e:
    logging.error("An error occurred during training: " + str(e), exc_info=True)
