from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### HELPERS

### Helper class for word indexing
SOS_token = 0 # Start of sentence
EOS_token = 1 # End of sentence

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2 # Initialize w/ SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            # Add new word
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            # Add seen word by increasing its count
            self.word2count[word] += 1

### Normalize text
def unicode_to_ascii(s):
    # Convert Unicode string to plain ASCII characters
    normalized_s = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
    return ''.join(normalized_s)

def normalize_string(s):
    # Lowercase, strip whitespace, remove punctuation and non-alphabet characters
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


### Parse and clean text data
def readLangs(lang1, lang2, reverse=False):

    print('Reading lines from file...')

    # Read text from file, split into lines
    data_file = 'jokes.tsv'
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')

    # Split lines into pairs, normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse: # If we're reversing pairs
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


##### PREPROCESSING
MAX_LENGTH = 25 # Max sentence length, number of words

def pair_filter(p):
    """
    Filter for pairs that fall within the MAX_LENGTH and start with our prefixes
    Returns True or False

    If X to eng/reverse=True -> p[1].startswith
    If eng to X/reverse=False -> p[0].startswith
    """
    return (len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH)

def filter_pairs(pairs):
    # Apply pair filter
    return [pair for pair in pairs if pair_filter(pair)]

### Prepare data
def prepare_data(lang1, lang2, reverse=False):
    # Read sentence pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print('Read %s sentence pairs' % len(pairs))

    # Filter pairs
    pairs = filter_pairs(pairs)
    print('Filtered down to %s sentence pairs' % len(pairs))

    # Count words
    print('Counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

# Sample pairs
input_lang, output_lang, pairs = prepare_data('jokes', 'punchlines', False)
print(random.choice(pairs))


##### SEQ2SEQ MODEL

class EncoderRNN(nn.Module):
    """
    Seq2seq encoder is an RNN.

    For each input word, the encoder outputs a vector and a hidden state, and
    uses the hidden state for the next input word.
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    """
    Decoder is another RNN that takes in the encoder output vector(s) and
    outputs a sequence of words to create the translation.

    The most basic seq2seq decoder uses only the last output of the encoder.
    This last output is sometimes caled the "context vector", as it encodes
    the context of the entire sequence. This context vector is used as the
    initial hidden state of the decoder.

    At each step of decoding, the decoder is given an input token and hidden
    state. The initial input token is the start of string (SOS) token.
    The first hidden state is the context vector (the encoder's last hidden
    state).
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


##### ATTENTION
"""
Calculate a set of attention weights.

Multiply attention weights by the encoder output vectors to create a weighted
combination. The result would contain information about that specific part of
the input sequence, and thus help the decoder choose the right output words.

To calculate the attention weights, we'll use a feed-forward layer that uses
the decoder's input and hidden state as inputs.

We will have to choose a max sentence length (input length, for encoder outputs),
wherein sentences of the max length will use all attention weights, while shorter
sentences would only use the first few.
"""

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attention_applied = torch.bmm(attention_weights.unsqueeze(0),
                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


##### NETWORK PREPROCESSING HELPERS
"""
Prepare training data by converting pairs into input and target tensors.
"""

def indices_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indices = indices_from_sentence(lang, sentence)
    indices.append(EOS_token)
    sentence_tensor = torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    return sentence_tensor

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])

    return (input_tensor, target_tensor)


##### DISPLAY HELPERS
"""
Helper functions for printing time elapsed and estimated remaining time for
training.
"""
import time
import math

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


##### MODEL TRAINING
"""
Training:
    - Run input sentence through encoder
    - Keep track of every output and the last hidden state
    - Decoder is given the start of sentence token (SOS) as its first input, and
        the last hidden state of the encoder as its first hidden state.

Teacher forcing ratio:
    - Teacher forcing uses real target outputs as each next input, rather than
        the decoder's guess as the next input. More teacher forcing -> faster
        convergence, at the tradeoff of potential instability.
    - Ratio means we randomly choose whether or not to use teacher forcing.
"""

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Train one interation
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode input
    for e_i in range(input_length):
        # Include hidden state from the last input when encoding current input
        encoder_output, encoder_hidden = encoder(input_tensor[e_i], encoder_hidden)
        encoder_outputs[e_i] = encoder_output[0, 0]

    # Decoder uses SOS token as first input
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Decoder uses last hidden state of encoder as first hidden state
    decoder_hidden = encoder_hidden

    # Randomly decide whether or not to use teacher forcing for decoder
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for d_i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[d_i])

            decoder_input = target_tensor[d_i] # Teacher forcing
    else:
        # No teacher forcing: use decoder's prediction as next input
        for d_i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_v, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze().detach() # Detach from history as input

            loss += criterion(decoder_output, target_tensor[d_i])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

"""

"""
def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """
    Train the network, track progress:
        - Start timer
        - Initialize optimizers and criterion
        - Create set of training pairs
        - Start empty losses array for plotting
        - Train many iterations, occasionally print progress and average loss.
    """

    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset after each print_every
    plot_loss_total = 0 # Reset after each plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss() # Negative log likelihood loss

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        # Print progress
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0 # Reset
            print('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters),
                             i, i / n_iters * 100, print_loss_avg))

        # Plot progress
        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0 # Reset

    show_plot(plot_losses)


##### PLOTTING RESULTS
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()

    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    plt.plot(points)
    # TODO: savefig


##### EVALUATION
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for e_i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[e_i], encoder_hidden)
            encoder_outputs[e_i] += encoder_output[0, 0]

        # Start of sentence token
        decoder_input = torch.tensor([[SOS_token]], device=device)

        # Decoder's initial hidden state is encoder's last hidden state
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for d_i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[d_i] = decoder_attention.data

            top_v, top_i = decoder_output.data.topk(1)

            if top_i.item() == EOS_token: # End of sentence
                decoded_words.append('<EOS>')
                break
            else:
                # Append prediction
                decoded_words.append(output_lang.index2word[top_i.item()])

            # Use prediction as input
            decoder_input = top_i.squeeze().detach()

        return decoded_words, decoder_attentions[:d_i + 1]


def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)

        print('>', pair[0])
        print('=', pair[1])

        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        print('<', output_sentence)
        print()


##### TRAIN AND EVALUATE
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attention_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

n_iters = 75000
train_iters(encoder, attention_decoder, n_iters, print_every=5000)

evaluate_randomly(encoder, attention_decoder)


### Visualizing Attention
test_phrase = 'why did the cookie go to the hospital ?'

output_words, attentions = evaluate(encoder, attention_decoder, test_phrase)
plt.matshow(attentions.numpy())

def show_attention(input_sentence, output_words, attentions):
    # TODO savefig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(encoder, attention_decoder, input_sentence)
    print('input =', input_sentence)
    print('output = ', ' '.join(output_words))

    show_attention(input_sentence, output_words, attentions)


# Test cases
evaluate_and_show_attention('what do lawyers wear to court ?')
evaluate_and_show_attention('why was the baby strawberry crying ?')
evaluate_and_show_attention('did you hear about the hungry clock ?')

evaluate_and_show_attention('why did the chicken cross the road ?')
evaluate_and_show_attention('who s there ?')
evaluate_and_show_attention('why are frogs so happy ?')
evaluate_and_show_attention('what is red and bad for your teeth ?')

# Save model
print('Saving model...')
torch.save(encoder, 'encoder_joke_punchline.pt')
torch.save(attention_decoder, 'attention_decoder_joke_punchline.pt')
