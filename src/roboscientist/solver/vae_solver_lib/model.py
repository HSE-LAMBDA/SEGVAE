from . import config

import torch
import torch.nn as nn

import numpy as np

from collections import namedtuple


ModelParams = namedtuple(
    'ModelParams', [
        'vocab_size', 'token_embedding_dim', 'hidden_dim', 'encoder_layers_cnt',
        'decoder_layers_cnt', 'latent_dim', 'device', 'x_dim'])
ModelParams.__new__.__defaults__ = (None,) * len(ModelParams._fields)


class FormulaVARE(nn.Module):
    # VAE Architecture is based on https://github.com/shentianxiao/text-autoencoders
    def __init__(self, model_params, ind2token, token2ind, condition=True):
        super().__init__()

        self.is_condition = condition
        if self.is_condition:
            # condition
            self.condition_encoder = nn.Linear(model_params.x_dim + 1, model_params.hidden_dim)
            self.condition_decoder = nn.Linear(model_params.x_dim + 1, model_params.hidden_dim)

        self.encoder = nn.LSTM(model_params.token_embedding_dim, model_params.hidden_dim,
                               model_params.encoder_layers_cnt, dropout=0, bidirectional=True)

        self.decoder = nn.LSTM(model_params.token_embedding_dim, model_params.hidden_dim,
                               model_params.decoder_layers_cnt, dropout=0)

        self.embedding = nn.Embedding(model_params.vocab_size, model_params.token_embedding_dim)
        self.linear = nn.Linear(model_params.hidden_dim, model_params.vocab_size)
        self.drop = nn.Dropout(0.1)

        self.hidden_to_mu = nn.Linear(model_params.hidden_dim * 2, model_params.latent_dim)
        self.hidden_to_logsigma = nn.Linear(model_params.hidden_dim * 2, model_params.latent_dim)
        self.z_to_embedding = nn.Linear(model_params.latent_dim, model_params.token_embedding_dim)

        self.latent_dim = model_params.latent_dim
        self.hidden_dim = model_params.hidden_dim

        self.encoder_layers_cnt = model_params.encoder_layers_cnt
        self.decoder_layers_cnt = model_params.decoder_layers_cnt

        self.device = model_params.device

        self._ind2token = ind2token
        self._token2ind = token2ind

        # self._reset_parameters()

    @staticmethod
    def sample_z(mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def encode(self, tokens, X, y):
        """
        :param tokens: input sequence (formula_len, batch_size)
        :param X: X_dataset for each formula in tokens (batch_size, n_points, x_dim)
        :param y: y_dataset for each formula in tokens (y_i = f_i(X_i)) (batch_size, n_points, 1)
        :return: mu (batch_size, latent_dim), logsigma (batch_size, latent_dim)
        """
        # print('X', X.shape)
        # print('y', y.shape)
        if y.shape[-1] != 1:
            for i in range(y.shape[0]):
                print(y[i].shape)
        if self.is_condition:
            condition = torch.from_numpy(np.concatenate((X, y), axis=-1).astype(np.float32)).to(self.device)
            # condition: (batch_size, n_points, x_dim + 1)
            hidden = self.condition_encoder(condition).mean(dim=1)
            # hidden: (batch_size, hidden_dim)
            hidden = torch.repeat_interleave(hidden.unsqueeze(0), 2 * self.encoder_layers_cnt, dim=0)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # tokens: (formula_len, batch_size)
        tokens = self.embedding(tokens)
        # tokens: (formula_len, batch_size, embedding_dim)
        tokens = self.drop(tokens)
        # hidden_state: (formula_len, batch_size, hidden_dim)
        if self.is_condition:
            c = torch.zeros_like(hidden).to(self.device)
            _, (hidden_state, _) = self.encoder(tokens, (hidden, c))
        else:
            _, (hidden_state, _) = self.encoder(tokens)
        hidden_state = torch.cat([hidden_state[-2], hidden_state[-1]], 1)
        mu = self.hidden_to_mu(hidden_state)
        # mu: (batch_size, latent_dim)
        logsigma = self.hidden_to_logsigma(hidden_state)
        # logsigma: (batch_size, latent_dim)
        return mu, logsigma

    def decode(self, tokens, z, hidden=None):
        """
        Latent into logits
        :param tokens: input sequence (formula_len, batch_size)
        :param z: latent vectors
        :param hidden: will be passed to LSTM decoder
        :return: logits (formula_len, batch_size, vocab_size), hidden
        """
        # tokens: (formula_len, batch_size)
        z_emb = self.z_to_embedding(z)
        tokens = self.embedding(tokens)
        tokens = self.drop(tokens)
        # tokens: (formula_len, batch_size, embedding_dim)
        tokens = tokens + z_emb
        # tokens: (formula_len, batch_size, embedding_dim)
        tokens, hidden = self.decoder(tokens, hidden)
        # tokens: (formula_len, batch_size, embedding_dim)
        tokens = self.drop(tokens)
        logits = self.linear(tokens)
        # logits: (formula_len, batch_size, vocab_size)
        return logits, hidden

    def forward(self, tokens, Xs, ys):
        """
        :param tokens: input sequence (formula_len, batch_size)
        :param Xs: X_dataset for each formula in tokens (batch_size, n_points, x_dim)
        :param ys: y_dataset for each formula in tokens (y_i = f_i(X_i)) (batch_size, n_points, 1)
        :return: logits, mu, logsigma, z
        """
        mu, logsigma = self.encode(tokens, Xs, ys)
        z = self.sample_z(mu, logsigma)
        # z: (batch_size, latent_dim)
        if self.is_condition:
            condition = torch.from_numpy(np.concatenate((Xs, ys), axis=-1).astype(np.float32)).to(self.device)
            # condition: (batch_size, n_points, x_dim + 1)
            hidden = self.condition_decoder(condition).mean(dim=1)
            hidden = torch.repeat_interleave(hidden.unsqueeze(0), 1 * self.encoder_layers_cnt, dim=0)
            c = torch.zeros_like(hidden).to(self.device)
            hidden = (hidden, c)

            logits, _ = self.decode(tokens, z, hidden)
        else:
            logits, _ = self.decode(tokens, z)
        # logits: (formula_len, batch_size, vocab_size)
        return logits, mu, logsigma, z

    def build_ordered_latents(self, batches, order, strategy):
        """
        :param batches: batches of token sequences
        :param order: order of batches
        :param strategy: decoding strategy:
                            'mu': use z = mu
                            'sample': sample z from N(mu, logsigma)
        :return:
        """
        assert strategy in ['mu', 'sample'], 'wrong strategy'
        z = []
        for inputs, X, y in batches:
            mu, logsigma = self.encode(inputs, X, y)
            if strategy == 'sample':
                zi = self.sample_z(mu, logsigma).detach().cpu().numpy()
            elif strategy == 'mu':
                zi = mu.detach().cpu().numpy()
            else:
                raise 42
            z.append(zi)
        # z_shape = (len(z), -1, z[0].shape[1])
        batch_size = z[0].shape[1]
        z = np.concatenate(z, axis=0)
        _, z = zip(*sorted(zip(order, z), key=lambda t: t[0]))
        z = np.array(list(z))
        i = 0
        new_z = []
        while i < len(z):
            new_z.append(torch.tensor(z[i: i + batch_size]))
            i += batch_size
        return new_z

    def reconstructed_formulas_from_encoded_formulas(self, encoded_formulas):
        """
        Substitute numbers in encoded formulas by corresponding tokens
        :param encoded_formulas: (total_formula_count, max_len)
        :return: reconstructed_formulas
        """
        reconstructed_formulas = []
        for e_formula in encoded_formulas:
            reconstructed_formulas.append([self._ind2token[id] for id in e_formula[1:]])
        reconstructed_formulas = [
            f[:f.index(config.END_OF_SEQUENCE)] \
                if config.END_OF_SEQUENCE in f else f for f in reconstructed_formulas]

        return np.asarray(reconstructed_formulas, dtype=np.object)

    def maybe_write_formulas(self, reconstructed_formulas, zs, out_file=None):
        """
        Write formulas to the file, if it is provided
        :param reconstructed_formulas: reconstructed formulas
        :param zs: latent vectors
        :param out_file: optional, name of file to write the formulas.
                         latents will be written to f'{out_file}z'
        :return:
        """
        if out_file is not None:
            with open(out_file, 'w') as f:
                f.write('\n'.join([' '.join(formula) for formula in reconstructed_formulas]))
            with open(f'{out_file}z', 'w') as f:
                for zi in zs:
                    for zi_k in zi:
                        f.write('%f ' % zi_k)
                    f.write('\n')

    def reconstruct(self, batches, order, max_len, out_file=None, strategy='sample', Xs=None, ys=None):
        """
        :param batches: batches of formulas to reconstruct
        :param order: order of batches
        :param max_len: max length of a formula
        :param out_file: optional. File to save the reconstructed formulas
        :param strategy: decode strategy:
                                'mu': use z=mu
                                'sample': z~N(mu, logsigma)
        :param Xs: X_dataset
        :param ys: y_dataset
        :return: reconstructed_formulas, zs
        """
        z = self.build_ordered_latents(batches, order, strategy=strategy)
        zs = [zi for batch_z in z for zi in batch_z]
        # z: (batches, z_in_batch, latent_dim)
        encoded_formulas = self.reconstruct_encoded_formulas_from_latent_batched(z, max_len, Xs=Xs, ys=ys)
        # encoded_formulas: (total_formula_count, max_len)
        reconstructed_formulas = self.reconstructed_formulas_from_encoded_formulas(encoded_formulas)
        self.maybe_write_formulas(reconstructed_formulas, zs, out_file)

        return reconstructed_formulas, zs

    def _reconstruct_encoded_formulas_from_latent(self, zs, max_len, explore=False, eps=0.2, Xs=None, ys=None,
                                                  sample=False):
        """
        :param zs: latents: (z_in_batch, latent_dim)
        :param max_len: max formula length
        :param explore: add exploration, currently unused
        :param eps: exploration rate, currently unused
        :param Xs: X_dataset
        :param ys: y_dataset
        :param sample: whether to sample from logits distribution
                                True: sample each next token from logits distribution
                                False: take the most probable token
        :return: formulas (encoded form: tokens as numbers)
        """
        formulas = []
        tokens = torch.zeros(1, len(zs), dtype=torch.long, device=self.device).fill_(
            self._token2ind[config.START_OF_SEQUENCE])
        # tokens: (1, z_in_batch)
        hidden = None
        if Xs is not None and self.is_condition:
            condition = torch.from_numpy(np.concatenate((Xs, ys), axis=-1).astype(np.float32)).to(self.device)
            # condition: (batch_size, n_points, x_dim + 1)
            hidden = self.condition_decoder(condition).mean(dim=1)
            hidden = torch.repeat_interleave(hidden.unsqueeze(0), 1 * self.encoder_layers_cnt, dim=0)
            c = torch.zeros_like(hidden).to(self.device)
            hidden = (hidden, c)
        for i in range(max_len):
            formulas.append(tokens)
            logits, hidden = self.decode(tokens, torch.tensor(zs, device=self.device), hidden)
            # logits: (formula_len, batch_size, vocab_size)
            if not sample:
                tokens = logits.argmax(dim=-1)
            else:
                tokens = torch.multinomial(torch.softmax(logits, dim=-1).squeeze(), 1).reshape(1, -1)
        # formulas_in_batch [[[f1_0, f2_0, ..]], [[f1_1, f2_1, ..]], ..] -> [[f1_0, f1_1, ..], [f2_0, f2_1, ..], ..]
        formulas = torch.cat(formulas, 0).T
        return formulas

    def reconstruct_encoded_formulas_from_latent_batched(self, z_batched, max_len, Xs=None, ys=None):
        """
        :param z_batched: z in batches
        :param max_len: max formula length
        :param Xs: X_dataset
        :param ys: y_dataset
        :return: formulas (encoded format: tokens are encoded as numbers)
        """
        # z_batched: (batches, z_in_batch, latent_dim)
        formulas = []
        for z in z_batched:
            formulas_in_batch = self._reconstruct_encoded_formulas_from_latent(z, max_len, Xs=Xs, ys=ys)
            for f in formulas_in_batch:
                formulas.append(f)
        return formulas

    def sample(self, n_formulas, max_len, out_file=None, ensure_valid=True, unique=True, Xs=None, ys=None,
               sample_from_logits=False, zs=None):
        """
        :param n_formulas: number of formulas to sample
        :param max_len: max formula length
        :param out_file: file to write formulas to
        :param ensure_valid: return only valid sampled formulas.
                             If a formula prefix is a valid formula -> substitute the formula by it
                             If formula and each its prefix is an invalid formula: do not return it
        :param unique: whether to return only unique sampled formulas
        :param Xs: X_dataset
        :param ys: y_dataset
        :param sample_from_logits: if true, formulas will be sampled from logits distribution. Otherwise,
                                   the most probable tokens will be chosen
        :param zs: Latent vectors. If not provided, zs will be sampled from N(0, 1)
        :return: (reconstructed_formulas,
                  zs,
                  n_formulas_sampled,
                  n_valid_formulas_sampled,
                  n_unique_valid_formulas_sampled)
        """
        if zs is None:
            zs = np.random.normal(size=(n_formulas, self.latent_dim)).astype('f')
        encoded_formulas = self._reconstruct_encoded_formulas_from_latent(zs, max_len, Xs=Xs, ys=ys,
                                                                          sample=sample_from_logits)
        reconstructed_formulas = self.reconstructed_formulas_from_encoded_formulas(encoded_formulas)

        n_formulas_sampled = len(reconstructed_formulas)

        if ensure_valid:
            valid_formulas = []
            reconstructed_formulas = valid_formulas
            pass

        n_valid_formulas_sampled = len(reconstructed_formulas)

        if unique:
            reconstructed_formulas = np.unique(reconstructed_formulas)
        self.maybe_write_formulas(reconstructed_formulas, zs, out_file)

        n_unique_valid_formulas_sampled = len(reconstructed_formulas)

        return reconstructed_formulas, zs, n_formulas_sampled, n_valid_formulas_sampled, n_unique_valid_formulas_sampled

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
