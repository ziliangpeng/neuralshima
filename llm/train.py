import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import argparse
from tqdm import tqdm

# Define a custom dataset
class CustomTextDataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_length=512):
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(folder_path)

    def load_data(self, folder_path):
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data.extend(f.read().split('\n'))
        return [text for text in data if text.strip()]  # Remove empty lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    return input_ids, attention_mask

def main(data_folder):
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set up dataset and dataloader
    dataset = CustomTextDataset(data_folder, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Generate a sample text after each epoch
        model.eval()
        with torch.no_grad():
            sample_input = tokenizer.encode("The quick brown fox", return_tensors="pt").to(device)
            sample_output = model.generate(sample_input, max_length=50, num_return_sequences=1, temperature=0.7)
            print("Sample generated text:")
            print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
            print()

    # Save the trained model
    model.save_pretrained('path/to/save/trained/model')
    tokenizer.save_pretrained('path/to/save/trained/model')

    print("Training completed and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model on a folder of text files.")
    parser.add_argument("data_folder", type=str, help="Path to the folder containing training data (.txt files)")
    args = parser.parse_args()
    
    main(args.data_folder)
