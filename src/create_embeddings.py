import numpy as np
from pathlib import Path
from datetime import date
import argparse
import shutil

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
    data_dir = base_dir / 'data' / 'raw'
    syllable_level_dir = data_dir / 'syllable_level_npy_39'
    word_level_dir = data_dir / 'word_level_29'

    def __init__(self, ngram=3):
        # initialize the dataset by creating SkipGram Data
        # 1. Load in all the files
        # 2. Fetch out the lyrics from it
        # 3. Create n-grams as required

        vocab = set()
        ngrams = []

        logger.info("Reading syllable level files")
        syllable_f_names = self.syllable_level_dir.iterdir()
        for i, f_name in enumerate(syllable_f_names):
            f_data = np.load(f_name, allow_pickle=True)
            lyrics = f_data[0][2]
#             lyrics = lyrics[:100]
            f_ngrams = self.generate_ngrams(lyrics, ngram)
            ngrams.extend(f_ngrams)
            vocab = vocab.union(lyrics)
#             if i == 10:
#                 break
            if (i+1) % 1000 == 0:
                logger.info("Completed reading {} syllable level files in dataloader".format(i+1))
#             break

        logger.info("Reading word level files")
        word_f_names = self.word_level_dir.iterdir()
        for i, f_name in enumerate(word_f_names):
            f_data = np.load(f_name, allow_pickle=True)
            lyrics = f_data[0][2]
            lyrics = [w[0] for w in lyrics]
            f_ngrams = self.generate_ngrams(lyrics, ngram)
            ngrams.extend(f_ngrams)
            vocab = vocab.union(lyrics)
#             if i == 10:
#                 break
            if (i+1) % 1000 == 0:
                logger.info("Completed reading {} word level files in dataloader")

        self.ngrams = ngrams
        self.vocab = sorted(vocab)
        self.vocab_size = len(self.vocab)
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}

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


def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir / 'skipgram_embeddings_checkpoint.pt'
    logger.info("Saving checkpoint to {}".format(f_path))
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / 'skipgram_embeddings_best_model.pt'
        logger.info("Saving checkpoint as best model")
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Checkpoint loaded successfully from {}".format(checkpoint_fpath))
    return model, optimizer, checkpoint['epoch']


def train(train_data_iterator, model, optimizer, criterion, start_epoch, epochs, loss_threshold, device, checkpoint_dir, model_dir, save_every, print_every):
    prev_loss = 100000000
    is_best = False
    to_break = False
    losses = []

    epochs = start_epoch + epochs

    for epoch in range(start_epoch, epochs):
        model.train()
        if (epoch + 1) % print_every == 0:
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

        losses.append(total_loss)
        logger.info("Loss is : {}".format(total_loss))

        if (epoch+1) % print_every == 0:
            print("Loss is : {}".format(total_loss))

        if (epoch+1) % save_every == 0:
            loss_change = prev_loss - total_loss
            logger.info("Change in loss after {} epochs is: {}".format(save_every, loss_change))
            if loss_change > 0:
                is_best=True
            if loss_change < loss_threshold:
                to_break=True
                
            prev_loss = total_loss
            
            logger.info("Creating checkpoint at epoch {}".format(epoch+1))
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, is_best, checkpoint_dir, model_dir)
            logger.info("Checkpoint created")

        if to_break:
            logger.info("Change in loss is less than the threshold. Stopping training")
            break

    logger.info("Completed Training")


def init_config(level, name, filename=None):
    logger.Logger(level, name, filename=filename)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--checkpoint_fname', type=str, default=None)

    args = argparser.parse_args()

    filepath = Path(__file__).absolute()
    base_dir = filepath.parents[1]
    model_dir = base_dir / 'model'
    out_dir = base_dir / 'output'
    log_dir = base_dir / 'logs'
    checkpoint_dir = base_dir / 'data' / 'model_checkpoint'
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    today_date = str(date.today())

    init_config("info", "creating embeddings", log_dir / 'creating_embeddings.log')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    logger.info("Using {} device".format(device))

    # Dataloader params
    data_params = {'batch_size': 20000,
                   'shuffle': True,
                   'num_workers': 4}
    logger.info("Data Parameters used are: {}".format(data_params))

    # Model params
    ngrams = 3
    context_size = ngrams - 1
    embedding_dim = 128
    hidden_dim = 256
    logger.info("Model Parameters used are: NGrams-{}, ContextSize-{}, EmbeddingsDim-{}, HiddenDim-{}".format(ngrams, context_size, embedding_dim, hidden_dim))

    # Training params
    start_epoch = 0
    epochs = 50000
    loss_threshold = 0.001
    learning_rate = 0.001
    logger.info("Training Parameters are: Epochs-{}, LossThreshold-{}, LearningRate-{}".format(epochs, loss_threshold, learning_rate))

    save_every = 10
    print_every = 10
    logger.info("Saving Checkpoint every {} epochs".format(save_every))
    logger.info("Printing Epoch Loss every {} epochs".format(print_every))

    logger.info("Creating the dataloader")
    training_set = LyricsNGramsDataset(ngrams)
    train_data_iterator = data.DataLoader(training_set, **data_params)

    vocab_size = training_set.vocab_size
    logger.info("Vocabulary size is: {}".format(vocab_size))

    vocab_lookup = training_set.word_to_ix
    vocab_fname = '{}_vocabulary_lookup.json'.format(today_date)
    vocab_fpath = out_dir / vocab_fname
    logger.info("Saving vocabulary lookup to {}".format(vocab_fpath))
    with open(vocab_fpath, 'w') as fp:
        json.dump(vocab_lookup, fp)

    logger.info("Initializing the model")
    model = LyricsEmbeddings(vocab_size, embedding_dim, context_size, hidden_dim)
    logger.info("Initializing the optimizer")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if args.checkpoint_fname:
        logger.info("Initializing the model and optimizer")
        logger.info("Loading model from the checkpoint")
        checkpoint_fpath = checkpoint_dir / args.checkpoint_fname
        model, optimizer, start_epoch = load_checkpoint(checkpoint_fpath, model, optimizer)
        
        logger.info("Moving optimizer buffer to Device as well.")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
    logger.info("{}".format(model))

    logger.info("Transfering model to {}".format(device))
    model = model.to(device)

    logger.info("Initilizing the loss criterion")
    criterion = LossCompute()

    logger.info("Entering the training loop")
    train(train_data_iterator, model, optimizer, criterion, start_epoch, epochs, loss_threshold, device, checkpoint_dir, model_dir, save_every, print_every)

    embedding_vec = model.embeddings.weight.data

    model_fname = '{}_skipgram_embeddings_entire_model.pt'.format(today_date)
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

    logger.info("Completed creating embeddings")


if __name__ == '__main__':
    main()
