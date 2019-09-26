import numpy as np
from pathlib import Path
from datetime import date

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils import data
import json

import src.utils.logger as logger


class LyricsNGramsDataset(data.Dataset):
    filepath = Path(__file__).absolute()
    base_dir = filepath.parents[1]
    data_dir = base_dir / 'data' / 'raw' / 'lmd-full_and_reddit_MIDI_dataset'
    syllable_level_dir = data_dir / 'syllable_level_npy_39'

    def __init__(self, ngram=3):
        # initialize the dataset by creating SkipGrap Data
        # 1. Load in all the files
        # 2. Fetch out the lyrics from it
        # 3. Create n-grams as required
        f_names = self.syllable_level_dir.iterdir()
        vocab = set()
        ngrams = []
        for i, f_name in enumerate(f_names):
            f_data = np.load(f_name, allow_pickle=True)
            lyrics = f_data[0][2]
            lyrics = lyrics[:100]
            f_ngrams = self.generate_ngrams(lyrics, ngram)
            ngrams.extend(f_ngrams)
            vocab = vocab.union(lyrics)
            if i%999 == 0:
                logger.info("Completed reading {} files in dataloader".format(i+1))
            break

        self.ngrams = ngrams
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.word_to_ix = {word: i for i, word in enumerate(vocab)}

        logger.info("Creating n-grams")
        idx_ngrams = [[self.word_to_ix[w] for w in ngram] for ngram in ngrams]
        self.idx_ngrams = [[ngram[:-1], ngram[-1]] for ngram in idx_ngrams]

    def generate_ngrams(self, word_lst, n):
        # Use the zip function to help us generate n-grams
        # Return a list of tuples
        # Each tuple is (word_i-2, word_i-1, word_i)
        ngrams = zip(*[word_lst[i:] for i in range(n)])
        return [ngram for ngram in ngrams]

    def __len__(self):
        return len(self.idx_ngrams)

    def __getitem__(self, i):
        context, target = self.idx_ngrams[i]
        context = torch.tensor(context, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return context, target


class LyricsEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128):
        super(LyricsEmbeddings, self).__init__()

        # matrix to keep the embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # print(inputs)
        # check why this view is needed!
        embeds = self.embeddings(inputs)
        # print(embeds)
        # print(embeds.shape)
        embeds = embeds.view((embeds.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)

        log_probab = F.log_softmax(out, dim=1)
        return log_probab


class LossCompute(object):
    def __init__(self):
        self.criterion = nn.NLLLoss()

    def __call__(self, x, y):
        """
        Call to compute loss
        :param x: predicted value
        :param y: actual value
        :return:
        """
        loss = self.criterion(x, y)
        return loss


def train(train_data_iterator, model, optimizer, criterion, epochs, device):
    losses = []
    for epoch in range(epochs):
        model.train()
        print("Running epoch {} / {}".format(epoch+1, epochs))
        logger.info("Running epoch {} / {}".format(epoch + 1, epochs))
        total_loss = 0

        for num_steps, data in enumerate(train_data_iterator):
            context = data[0].to(device)
            target = data[1].to(device)

            optimizer.zero_grad()

            # print(context)
            log_probabs = model(context)

            loss = criterion(log_probabs, target)
            # print(loss)
            # print(type(loss))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if num_steps%10 == 0:
                logger.info("Loss at Step {} is {}".format(num_steps+1, loss.item()))

        losses.append(total_loss)
        logger.info("Loss is : {}".format(total_loss))
        print("Loss is : {}".format(total_loss))
    logger.info("Completed Training")


def init_config(level, name, filename=None):
    logger.Logger(level, name, filename=filename)


def main():
    filepath = Path(__file__).absolute()
    base_dir = filepath.parents[1]
    model_dir = base_dir / 'model'
    out_dir = base_dir / 'output'
    log_dir = base_dir / 'logs'
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    init_config("info", "creating embeddings", log_dir / 'creating_embeddings.log')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    logger.info("Using {} device".format(device))

    # Dataloader params
    data_params = {'batch_size': 100,
                   'shuffle': True,
                   'num_workers': 4}
    logger.info("Data Parameters used are: {}".format(data_params))

    # Model params
    ngrams = 3
    context_size = ngrams - 1
    embedding_dim = 64
    hidden_dim = 128
    logger.info("Model Parameters used are: NGrams-{}, ContextSize-{}, EmbeddingsDim-{}, HiddenDim-{}".format(ngrams, context_size, embedding_dim, hidden_dim))

    # Training params
    epochs = 4
    learning_rate = 0.001
    logger.info("Training Parameters are: Epochs-{}, LearningRate-{}".format(epochs, learning_rate))

    logger.info("Creating the dataloader")
    training_set = LyricsNGramsDataset(ngrams)
    train_data_iterator = data.DataLoader(training_set, **data_params)

    vocab_size = training_set.vocab_size
    logger.info("Vocabulary size is: {}".format(vocab_size))

    logger.info("Initializing the model")
    model = LyricsEmbeddings(vocab_size, embedding_dim, context_size, hidden_dim)
    logger.info("{}".format(model))

    logger.info("Transfering model to {}".format(device))
    model = model.to(device)

    logger.info("Initializing the optimizer")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    logger.info("Initilizing the loss criterion")
    criterion = LossCompute()

    logger.info("Entering the training loop")
    train(train_data_iterator, model, optimizer, criterion, epochs, device)

    embedding_vec = model.embeddings.weight.data
    vocab_lookup = training_set.word_to_ix

    today_date = str(date.today())

    model_fname = '{}_embeddings_entire_model.pt'.format(today_date)
    model_fpath = model_dir / model_fname
    logger.info("Saving model state dict to {}".format(model_fpath))
    torch.save(model.state_dict(), model_fpath)

    # Loading the model
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    embeddings_fname = '{}_embeddings_vector.pt'.format(today_date)
    embeddings_fpath = out_dir / embeddings_fname
    logger.info("Saving embeddings tensor to {}".format(embeddings_fpath))
    torch.save(embedding_vec, embeddings_fpath)
    # Loading the embeddings
    # embeddings_vec = torch.load(out_dir / embeddings_fname)

    vocab_fname = '{}_vocabulary_lookup.json'.format(today_date)
    vocab_fpath = out_dir / vocab_fname
    logger.info("Saving vocabulary lookup to {}".format(vocab_fpath))
    with open(vocab_fpath, 'w') as fp:
        json.dump(vocab_lookup, fp)


if __name__ == '__main__':
    main()
