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


class Dataloader(data.Dataset):
    file_path = Path(__file__).absolute()
    base_dir = file_path.parents[1]
    embeddings_dir = base_dir / 'output'
    raw_data_dir = base_dir / 'data' / 'raw' / 'syllable_level_npy_39'

    def read_file(self, f_path):
        f_data = np.load(f_path, allow_pickle=True)
        cont_attributes = f_data[0][0][:100]
        discrete_attributes = f_data[0][1][:100]
        # lyrics = f_data[0][2]
        lyrics = f_data[0][2][:100]

        # print(type(cont_attributes))
        # print(cont_attributes)
        return cont_attributes, discrete_attributes, lyrics

    def convert_lyrics_to_ix(self, lyrics):
        lyrics_ix = [self.word_to_ix[i] for i in lyrics]
        return lyrics_ix

    def convert_lyrics_to_embeddings(self, lyrics):
        lyrics_embeddings = [self.embeddings_vec[self.word_to_ix[i]] for i in lyrics]
        return lyrics_embeddings

    def generate_ngrams(self, lst, n):
        # Use the zip function to help us generate n-grams
        # Return a list of tuples
        # Each tuple is (word_i-2, word_i-1, word_i)
        ngrams = zip(*[lst[i:] for i in range(n)])
        return [list(ngram) for ngram in ngrams]

    # def convert_lst_tensor_to_tensor(self, lst_tensor):
    #     out_tensor = torch.Tensor(len(lst_tensor), lst_tensor[0].shape[0])
    #     print(out_tensor)
    #     torch.cat(lst_tensor, out = out_tensor)
    #     return out_tensor
    #
    # def convert_lyrics_seq_to_tensor(self, lyrics_seq):
    #     # print(lyrics_seq)
    #     print(lyrics_seq[0])


    def create_melody_seq(self, cont_attr, discrete_attr, lyrics):
        """
        Takes in 3 lists and creates sequences out of it!
        :param cont_attr:
        :param discrete_attr:
        :param lyrics:
        :return:
        """
        lyrics_seq = self.generate_ngrams(lyrics, self.seq_len)
        # self.convert_lyrics_seq_to_tensor(lyrics_seq)
        # print(lyrics_seq)

        # print("Length of cont attributes")
        # print(len(cont_attr))
        cont_attr_seq = self.generate_ngrams(cont_attr, self.seq_len)
        # print(cont_attr_seq)

        discrete_attr = self.generate_ngrams(discrete_attr, self.seq_len)
        # seq = zip(*[lyrics_seq, cont_attr_seq, discrete_attr])
        return lyrics_seq, cont_attr_seq, discrete_attr

    def create_training_data(self):
        # all_seq is a list of all the sequences.
        # This might explode when dealing with the entire data.
        # Might need an alternate way out!!
        # TODO: Check this with entire data
        all_lyrics_seq = []
        all_cont_attr_seq = []
        all_discrete_attr_seq = []
        f_names = self.raw_data_dir.iterdir()
        for i, f_name in enumerate(f_names):
            cont_attr, discrete_attr, lyrics = self.read_file(f_name)
            # print(lyrics)

            # TODO: Remove creating lyrics and the function if not being used below!
            lyrics_ix = self.convert_lyrics_to_ix(lyrics)
            # print(lyrics_ix)
            lyrics_embeddings = self.convert_lyrics_to_embeddings(lyrics)
            # print(type(lyrics_embeddings))

            # TODO: Decide to what to use here. Embeddings directly or just lyrics index
            lyrics_seq, cont_attr_seq, discrete_attr_seq = self.create_melody_seq(cont_attr, discrete_attr, lyrics_embeddings)

            # print("Printing here")
            # # print(len(discrete_attr_seq))
            # print(cont_attr_seq)
            # print(lyrics_seq)
            # print(discrete_attr_seq)

            # print(f_seq)
            all_lyrics_seq.extend(lyrics_seq)
            all_cont_attr_seq.extend(cont_attr_seq)
            all_discrete_attr_seq.extend(discrete_attr_seq)
            # TODO: Remove the break statement.
            break
        return all_lyrics_seq, all_cont_attr_seq, all_discrete_attr_seq

    def __init__(self, embeddings_fname, vocab_fname, seq_len):
        embeddings_vec = torch.load(self.embeddings_dir/ embeddings_fname)
        # TODO: Comment out the line below.
        embeddings_vec = embeddings_vec[:, :10]
        self.embeddings_vec = embeddings_vec.tolist()
        # print(self.embeddings_vec)
        with open(self.embeddings_dir / vocab_fname, 'r') as fp:
            self.word_to_ix = json.load(fp)
        # print(word_to_ix)

        self.seq_len = seq_len
        lyrics_seq, cont_attr_seq, discrete_attr_seq = self.create_training_data()

        # print(lyrics_seq)
        # print(cont_attr_seq)
        # print(discrete_attr)

        # print("Length of lyrics seq list: {}".format(len(lyrics_seq)))
        # print("Shape of one element: {}".format(len(lyrics_seq[0])))
        # print("An element is: {}".format(lyrics_seq[0]))

        # lyrics_seq_tensor = torch.Tensor(len(lyrics_seq), seq_len, 10)
        # torch.cat(lyrics_seq, out=lyrics_seq_tensor)
        # print(lyrics_seq_tensor[0])
        # self.lyrics_seq = lyrics_seq_tensor

        self.lyrics_seq = torch.Tensor(lyrics_seq)

        self.cont_attr_seq = torch.tensor(cont_attr_seq)

        # print(self.cont_attr_seq.shape)

        self.discrete_attr_seq = torch.tensor(discrete_attr_seq)
        # , ,  = self.create_training_data()

    def __len__(self):
        print(len(self.lyrics_seq))
        return len(self.lyrics_seq)

    def __getitem__(self, i):
        lyrics_seq = self.lyrics_seq[i]
        cont_val_seq = self.cont_attr_seq[i]
        discrete_val_seq = self.discrete_attr_seq[i]
        noise_seq = torch.rand(lyrics_seq.shape)

        return lyrics_seq, cont_val_seq, discrete_val_seq, noise_seq


class GeneratorLSTM(nn.Module):
    def __init__(self, embed_dim, ff1_out, hidden_dim, out_dim):
        super(GeneratorLSTM, self).__init__()

        self.input_ff = nn.Linear(embed_dim, ff1_out)
        self.lstm = nn.LSTM(ff1_out, hidden_dim, num_layers=2)
        self.output_ff = nn.Linear(ff1_out, out_dim)

    def forward(self, lyrics, noise):
        """
        Define forward pass
        1. Pass input (50 dim) through FF1 layer to explode the dimension
        2. Pass the entire sequence through
        :return:
        """
        # print("Input size {}".format(lyrics.shape))
        # Reshaping input is not required.
        # pytorch automatically applys the linear layer to only the last dimension!
        concat_lyrics = torch.cat((lyrics, noise), 2)
        print("Concat Shape: {}".format(concat_lyrics.shape))
        out1 = F.relu(self.input_ff(concat_lyrics))
        # print("Output size of first layer {}".format(out1.shape))
        # print(out1.shape)
        # lstm_out, _ = self.lstm(out1)
        # The input to LSTM needs to be reshaped.
        lstm_out, _ = self.lstm(out1.view(out1.shape[1], out1.shape[0], -1))
        # print(lstm_out.shape)

        tag = self.output_ff(lstm_out.view(lstm_out.shape[1], lstm_out.shape[0], -1))
        # print(tag.shape)

        # print(tag)
        return tag


class DiscriminatorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(DiscriminatorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, input, lyrics):
        concat_input = torch.cat((input, lyrics), 2)
        _, lstm_out = self.lstm(concat_input.view(concat_input.shape[1], concat_input.shape[0], -1))
        ct = lstm_out[1]
        # print(ct.shape)
        # print(ct[-1])
        last_layer_out = ct[-1]
        last_layer_out = last_layer_out.view(last_layer_out.shape[0], 1, -1)
        # print(last_layer_out.shape)
        out = torch.sigmoid(self.out(last_layer_out))
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        # print(out)


def train_conditional_gan():
    """
    Run training loop in epochs.
    In one epoch, have certain number of steps for which you optimize for Discriminator
    Then have one step for
    :return:
    """

    for epoch in range(start_epoch, epochs):
        discriminator.train()

        # if (epoch + 1) % print_every == 0:
        #     print("Running epoch {} / {}".format(epoch + 1, epochs))
        #
        # logger.info("Running epoch {} / {}".format(epoch + 1, epochs))
        total_loss = 0

        for num_steps, data in enumerate(train_data_iterator):
            lyrics_seq = data[0].to(device)
            cont_val_seq = data[1].to(device)
            discrete_val_seq = data[2].to(device)
            noise_seq = data[3].to(device)

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            gen_out = generator(lyrics_seq, noise_seq)
            fake_dis_out = discriminator(gen_out, lyrics_seq)
            true_dis_out = discriminator(discrete_val_seq, lyrics_seq)

            fake_loss = criterion_D(fake_dis_out, fake_val)
            true_loss = criterion_D(true_dis_out, true_val)

            # Read this. But whyyy?
            discriminator_loss = (fake_loss + true_loss) / 2
            discriminator_loss.backward()
            optimizer_D.step()
            # print(loss)
            # print(type(loss))

            # TODO - Check till here
            # TODO - Write code to train generator!
            # TODO - Happy Data Sciencing! <3

            total_loss += loss.item()

        losses.append(total_loss)
        logger.info("Loss is : {}".format(total_loss))

        if (epoch + 1) % print_every == 0:
            print("Loss is : {}".format(total_loss))

        if (epoch + 1) % save_every == 0:
            loss_change = prev_loss - total_loss
            logger.info(
                "Change in loss after {} epochs is: {}".format(save_every,
                                                               loss_change))
            if loss_change > 0:
                is_best = True
            if loss_change < loss_threshold:
                to_break = True

            prev_loss = total_loss

            logger.info("Creating checkpoint at epoch {}".format(epoch + 1))
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, is_best, checkpoint_dir, model_dir)
            logger.info("Checkpoint created")

        if to_break:
            logger.info(
                "Change in loss is less than the threshold. Stopping training")
            break

    logger.info("Completed Training")


if __name__ == '__main__':
    data_params = {'batch_size': 8,
                   'shuffle': True,
                   'num_workers': 1}

    sequence_len = 5
    training_set = Dataloader('2019-09-26_embeddings_vector.pt', '2019-09-26_vocabulary_lookup.json', sequence_len)
    train_data_iterator = data.DataLoader(training_set, **data_params)

    generator = GeneratorLSTM(20, 40, 40, 3)
    discriminator = DiscriminatorLSTM(13, 40, 1)

    for i, data in enumerate(train_data_iterator):
        lyrics_seq = data[0]
        cont_val_seq = data[1]
        discrete_val_seq = data[2]
        noise_seq = data[3]
        print("Lyrics sequence is: {}".format(lyrics_seq.shape))
        print("Content Value Sequence is: {}".format(cont_val_seq.shape))
        print("Discrete value sequence is: {}".format(discrete_val_seq.shape))

        print(len(lyrics_seq))

        # generator.zero_grad()
        # discriminator.zero_grad()
        gen_out = generator(lyrics_seq, noise_seq)

        discriminator(gen_out, lyrics_seq)

        break