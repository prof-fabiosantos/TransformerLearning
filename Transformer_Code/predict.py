
import torch
import re
from transformer import Transformer

# Re-using the Vocabulary class and normalize_string function from train.py
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

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def predict(question, model, question_vocab, answer_vocab, max_len=20):
    model.eval() # Set model to evaluation mode

    # Process the input question
    q_normalized = normalize_string(question)
    q_tokens = [question_vocab.word2index.get(word, 0) for word in q_normalized.split(' ')]
    q_tokens.append(question_vocab.word2index["<EOS>"])
    src = torch.tensor(q_tokens).unsqueeze(0)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        e_outputs = model.encoder(src, src_mask)

    # Start decoding with <SOS> token
    trg_indexes = [answer_vocab.word2index["<SOS>"]]

    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indexes).unsqueeze(0)
        trg_pad_mask = (trg_tensor != 0).unsqueeze(1).unsqueeze(2)
        trg_len = trg_tensor.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool))
        trg_mask = trg_pad_mask & trg_sub_mask

        with torch.no_grad():
            d_output = model.decoder(trg_tensor, e_outputs, src_mask, trg_mask)
            output = model.out(d_output)
            pred_token = output.argmax(2)[:, -1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == answer_vocab.word2index["<EOS>"]:
            break

    trg_tokens = [answer_vocab.index2word[i] for i in trg_indexes]
    return ' '.join(trg_tokens[1:-1]) # Remove <SOS> and <EOS>

if __name__ == "__main__":
    # --- Load Vocabularies ---
    with open("questions.txt", "r", encoding="utf-8") as f:
        questions = [normalize_string(line) for line in f]
    with open("answers.txt", "r", encoding="utf-8") as f:
        answers = [normalize_string(line) for line in f]

    question_vocab = Vocabulary()
    answer_vocab = Vocabulary()
    for q in questions:
        question_vocab.add_sentence(q)
    for a in answers:
        answer_vocab.add_sentence(a)

    # --- Load Model ---
    SRC_VOCAB_SIZE = question_vocab.n_words
    TRG_VOCAB_SIZE = answer_vocab.n_words
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    DROPOUT = 0.1

    model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, DROPOUT)
    model.load_state_dict(torch.load("transformer_qa.pth"))

    # --- Make a Prediction ---
    test_question = "Qual o maior planeta do sistema solar?"
    predicted_answer = predict(test_question, model, question_vocab, answer_vocab)
    
    print(f"Pergunta: {test_question}")
    print(f"Resposta: {predicted_answer}")
