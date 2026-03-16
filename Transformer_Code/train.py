
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer
import re

# 1. Vocabulary Class
class Vocabulary:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.word_count = {}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word_count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1

# 2. Dataset Class
class QADataset(Dataset):
    def __init__(self, questions, answers, question_vocab, answer_vocab):
        self.questions = questions
        self.answers = answers
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        
        q_tokens = [self.question_vocab.word2index[word] for word in question.split(' ')]
        a_tokens = [self.answer_vocab.word2index[word] for word in answer.split(' ')]
        
        q_tokens.append(self.question_vocab.word2index["<EOS>"])
        a_tokens.append(self.answer_vocab.word2index["<EOS>"])

        return torch.tensor(q_tokens), torch.tensor(a_tokens)

# Helper function to clean and normalize text
def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Collate function to pad sequences
def collate_fn(batch):
    questions, answers = zip(*batch)
    
    q_padded = nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    a_padded = nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=0)
    
    return q_padded, a_padded

if __name__ == "__main__":
    # Load and process data
    with open("questions.txt", "r", encoding="utf-8") as f:
        questions = [normalize_string(line) for line in f]
    with open("answers.txt", "r", encoding="utf-8") as f:
        answers = [normalize_string(line) for line in f]

    # Create vocabularies
    question_vocab = Vocabulary()
    answer_vocab = Vocabulary()
    for q in questions:
        question_vocab.add_sentence(q)
    for a in answers:
        answer_vocab.add_sentence(a)

    # Create dataset and dataloader
    dataset = QADataset(questions, answers, question_vocab, answer_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Model parameters
    SRC_VOCAB_SIZE = question_vocab.n_words
    TRG_VOCAB_SIZE = answer_vocab.n_words
    D_MODEL = 128 # smaller model for this small dataset
    N_LAYERS = 2
    N_HEADS = 4
    DROPOUT = 0.1

    # Instantiate model, loss function, and optimizer
    model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, DROPOUT)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    N_EPOCHS = 100
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        for src, trg in dataloader:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            trg_input = trg[:, :-1]
            trg_target = trg[:, 1:]

            trg_pad_mask = (trg_input != 0).unsqueeze(1).unsqueeze(2)
            trg_len = trg_input.shape[1]
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool))
            trg_mask = trg_pad_mask & trg_sub_mask

            optimizer.zero_grad()
            output = model(src, trg_input, src_mask, trg_mask)
            
            loss = criterion(output.view(-1, TRG_VOCAB_SIZE), trg_target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "transformer_qa.pth")
    print("Training complete. Model saved to transformer_qa.pth")
