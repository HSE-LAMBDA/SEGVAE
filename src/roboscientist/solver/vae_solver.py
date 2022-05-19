import roboscientist.solver.solver_base as rs_solver_base
import roboscientist.solver.vae_solver_lib.optimize_constants as rs_optimize_constants
import roboscientist.solver.vae_solver_lib.config as rs_config
import roboscientist.solver.vae_solver_lib.model as rs_model
import roboscientist.solver.vae_solver_lib.train as rs_train
import roboscientist.equation.equation as rs_equation
import roboscientist.equation.operators as rs_operators

from sklearn.metrics import mean_squared_error

import torch

from collections import deque, namedtuple
import numpy as np
import sympy as sp
import random


VAESolverParams = namedtuple(
    'VAESolverParams', [
        # problem parameters
        'true_formula',                             # Equation: true formula (needed for active learning)
        # model parameters
        'model_params',                             # Dict with model parameters. Must include: token_embedding_dim,
                                                    # hidden_dim, encoder_layers_cnt, decoder_layers_cnt, latent_dim,
                                                    # x_dim
        'is_condition',                             # is_condition
        'formula_predicate',                        # formula predicate

        # formula parameters
        'max_formula_length',                       # Int: Maximum length of a formula
        'max_degree',                               # Int: Max arity of a formula operator
        'functions',                                # List: A list of finctions used in formula
        # TODO(julia): remove arities
        'arities',                                  # Dict: A dict of arities of the functions.
                                                    # For each f in function arity must be provided
        'optimizable_constants',                    # List: Tokens of optimizable constants. Example: Symbol('const0')
        'float_constants',                          # List: a list of float constants used by the solver
        'free_variables',                           # List: a list of free variables used by the solver.
                                                    # Example: Symbol('x0')

        # training parameters
        'n_pretrain_steps',                         # Int: number of pretrain epochs (number of times the model will be
                                                    # trained on the fixed train dataset)
        'batch_size',                               # Int: batch size
        'n_pretrain_formulas',                      # Int: Number of formulas in pretrain dataset. If a train file is
                                                    # provided, this parameter will be ignored
        'create_pretrain_dataset',                  # Bool: Whether to create a pretrain dataset. If False, train
                                                    # dataset must  be provided. see: pretrain_train_file,
                                                    # pretrain_val_file
        'kl_coef',                                  # Float: Coefficient of KL-divergence in model loss
        'device',                                   # Device: cuda or cpu
        'learning_rate',                            # Float: learning rate
        'betas',                                    # Tuple(float, float): Adam parameter
        'retrain_strategy',                         # Str: retrain strategy:
                                                    # - "queue": use the best formulas (queue) generated so far to
                                                    # retrain the model
                                                    # - "last_steps": use the best formulas from the last
                                                    # |use_n_last_steps| to retrain the model
        'queue_size',                               # Int: the size of the queue to use, when using
                                                    # |retrain_strategy| == "queue"
        'use_n_last_steps',                         # Int: Use best formulas generated on last |use_n_last_steps| epochs
                                                    # for training and for percentile calculation
        'percentile',                               # Int: Use |percentile| best formulas for retraining
        'n_formulas_to_sample',                     # Int: Number of formulas to sample on each epochs
        'add_noise_to_model_params',                # Bool: Whether to add noise to model parameters
        'noise_coef',                               # Float: Noise coefficient.
                                                    # model weights = model weights + |noise_coef| * noise
        'add_noise_every_n_steps',                  # Int: Add noise to model on every |add_noise_every_n_steps| epoch
        'sample_from_logits',                       # Bool: If False -> most probable, True -> sample

        # files
        'retrain_file',                             # Str: File to retrain the model. Used for retraining stage
        'file_to_sample',                           # Str: File to sample formulas to. Used for retraining stage
        'pretrain_train_file',                      # Str: File with pretrain train formulas.
                                                    # If not |create_pretrain_dataset|, this will be used to pretrain
                                                    # the model. Otherwise generated pretrain dataset will be written
                                                    # to this file
        'pretrain_val_file',                        # Str: File with pretrain validation formulas.
                                                    # If not |create_pretrain_dataset|, this will be used to pretrain
                                                    # the model. Otherwise generated pretrain dataset will be written
                                                    #  to this file

        # specific settings
        'no_retrain',                               # Bool: if True, Don't retrain the model during the retraining phase
        'continue_training_on_pretrain_dataset',    # Bool: if True, continue training the model on the pretrain dataset

        # data
        'initial_xs',                               # numpy array: initial xs data
        'initial_ys',                               # numpy array: initial ys data

        # active learning
        'active_learning',                          # Bool: if True, active learning strategies will be used to
                                                    # increase the dataset
        'active_learning_epochs',                   # Int: do active learning every |active_learning_epochs| epochs
        'active_learning_strategy',                 # Str: active learning strategy
        'active_learning_n_x_candidates',           # Int: number of x candidates to consider when picking the next one
        'active_learning_n_sample',                 # Int: number of formulas to sample for active learning metric
                                                    # calculation
        'active_learning_file_to_sample',           # Srt: path to file to sample formulas to

        'const_opt_method',                         # Str: Type of constant optimizers. Should be one of "bfgs", "adam"

        'domains',                                  # TODO
        'simplification',                           # TODO
    ])

VAESolverParams.__new__.__defaults__ = (
    None,                                           # true_formula
    {'token_embedding_dim': 128, 'hidden_dim': 128,
     'encoder_layers_cnt': 1,
     'decoder_layers_cnt': 1, 'latent_dim':  8,
     'x_dim': 1},                                   # model_params
    False,                                          # is_condition
    lambda func: True,                              # formula_predicate
    15,                                             # max_formula_length
    2,                                              # max_degree
    ['sin', 'add', 'log'],                          # functions
    {'sin': 1, 'cos': 1, 'add': 2, 'log': 1},       # arities
    [],                                             # optimizable_constants
    [],                                             # float constants
    ["x1"],                                         # free variables
    50,                                             # n_pretrain_steps
    256,                                            # batch_size
    20000,                                          # n_pretrain_formulas
    False,                                          # create_pretrain_dataset
    0.2,                                            # kl_coef
    torch.device("cuda:0"),                         # device
    0.0005,                                         # learning_rate
    (0.5, 0.999),                                   # betas
    'last_steps',                                   # retrain_strategy
    256,                                            # queue_size
    5,                                              # use_n_last_steps
    20,                                             # percentile
    2000,                                           # n_formulas_to_sample
    False,                                          # add_noise_to_model_params
    0.01,                                           # noise_coef
    5,                                              # add_noise_every_n_steps
    False,                                          # sample_from_logits
    'retrain',                                      # retrain_file
    'sample',                                       # file_to_sample
    'train',                                        # pretrain_train_file
    'val',                                          # pretrain_val_file
    False,                                          # no_retrain
    False,                                          # continue_training_on_pretrain_dataset
    np.linspace(0.1, 1, 100),                       # initial_xs
    np.zeros(100),                                  # initial_ys
    False,                                          # active_learning
    1,                                              # active_learning_epochs
    'var',                                          # active_learning_strategy
    100,                                            # active_learning_n_x_candidates
    5000,                                           # active_learning_n_sample
    'active_learning_sample',                       # active_learning_file_to_sample
    'bfgs',                                         # const_opt_method
    None,                                           # domains TODO
    False,                                          # simplification TODO
)


class VAESolver(rs_solver_base.BaseSolver):
    def __init__(self, logger, checkpoint_file=None, solver_params=None):
        super().__init__(logger)

        if solver_params is None:
            solver_params = VAESolverParams()
        self.params = solver_params

        self._ind2token = self.params.functions + [str(c) for c in self.params.float_constants] + \
                          self.params.optimizable_constants + \
                          [rs_config.START_OF_SEQUENCE, rs_config.END_OF_SEQUENCE, rs_config.PADDING] + \
                          self.params.free_variables
        self._token2ind = {t: i for i, t in enumerate(self._ind2token)}

        if self.params.retrain_strategy == 'last_steps':
            self.stats = FormulaStatisticsLastN(use_n_last_steps=self.params.use_n_last_steps,
                                                percentile=self.params.percentile,
                                                simplification=self.params.simplification)
        if self.params.retrain_strategy == 'queue':
            self.stats = FormulaStatisticsQueue(self.params.queue_size)

        model_params = rs_model.ModelParams(vocab_size=len(self._ind2token), device=self.params.device,
                                         **self.params.model_params)
        self.model = rs_model.FormulaVARE(model_params, self._ind2token, self._token2ind,
                                       condition=self.params.is_condition)
        self.model.to(self.params.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate,
                                          betas=self.params.betas)

        self.xs = self.params.initial_xs.reshape(-1, self.params.model_params['x_dim'])
        self.ys = self.params.initial_ys

        self.const_opt_method = self.params.const_opt_method

        if checkpoint_file is not None:
            self._load_from_checkpoint(checkpoint_file)
        else:
            print("===== START PRETRAIN =====")
            self.pretrain_batches, _ = rs_train.build_ordered_batches(formula_file=self.params.pretrain_train_file,
                                                                      solver=self)
            self.valid_batches, _ = rs_train.build_ordered_batches(formula_file=self.params.pretrain_val_file,
                                                                   solver=self)
            rs_train.pretrain(n_pretrain_steps=self.params.n_pretrain_steps, model=self.model, optimizer=self.optimizer,
                           pretrain_batches=self.pretrain_batches, pretrain_val_batches=self.valid_batches,
                           kl_coef=self.params.kl_coef)
            print("===== END PRETRAIN =====")

    def log_metrics(self, reference_dataset, candidate_equations, all_constants, custom_log):
        if self._logger is not None:
            if not self.params.active_learning:
                self._logger.log_metrics(reference_dataset, candidate_equations, all_constants)
            else:
                self._logger.log_metrics(reference_dataset, candidate_equations, all_constants, self.xs, self.ys)
            self._logger.commit_metrics(custom_log)

    def create_checkpoint(self, checkpoint_file):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_file)

    def _load_from_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _training_step(self, reference_dataset, epoch):
        custom_log = {}
        self.stats.clear_the_oldest_step()

        # noises = self._maybe_add_noise_to_model_params(epoch)

        cond_x, cond_y = self._get_condition(self.params.n_formulas_to_sample)
        self.model.sample(self.params.n_formulas_to_sample, self.params.max_formula_length,
                          self.params.file_to_sample, Xs=cond_x, ys=cond_y, ensure_valid=False, unique=True,
                          sample_from_logits=self.params.sample_from_logits)

        # self._maybe_remove_noise_from_model_params(epoch, noises)

        valid_formulas = []
        valid_equations = []
        valid_mses = []
        all_constants = []
        n_all = 0
        n_false_predicate = 0
        n_invalid = 0
        n_optimize_failed = 0
        n_invalid_domain = 0
        with open(self.params.file_to_sample) as f:
            for line in f:
                n_all += 1
                # def isfloat(value):
                #     try:
                #         float(value)
                #         return True
                #     except ValueError:
                #         return False

                f_to_eval = line.strip().split()
                # f_to_eval = [float(x) if isfloat(x) else x for x in f_to_eval]
                f_to_eval = rs_equation.Equation(f_to_eval)
                if not f_to_eval.check_validity()[0]:
                    n_invalid += 1
                    continue

                if not self.params.formula_predicate(line.strip().split()):
                    n_false_predicate += 1
                    continue

                constants = rs_optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys, self.const_opt_method)
                if f_to_eval.const_count() > 0 and constants is None:
                    n_optimize_failed += 1

                lows, highs, y_dom = self.params.domains
                x_random = np.random.uniform(low=lows , high=highs,
                                             size=(len(self.params.free_variables) ** 2 * 10_000, 2))
                y_random = f_to_eval.func(x_random, constants)
                y = f_to_eval.func(self.xs.reshape(-1, self.params.model_params['x_dim']), constants)

                if np.any(np.isnan(y)) or np.any(np.isnan(y_random)):
                    n_invalid_domain += 1
                    continue
                if np.max(y_random) > y_dom[1] or np.min(y_random) < y_dom[0]:
                    n_invalid_domain += 1
                    continue

                if type(y) is float or y.shape == (1,) or y.shape == (1, 1) or y.shape == ():
                    y = np.repeat(np.array(y).astype(np.float64),
                                  self.xs.reshape(-1, self.params.model_params['x_dim']).shape[0]).reshape(-1, 1)
                mse = mean_squared_error(y, self.ys)
                valid_formulas.append(line.strip())
                valid_mses.append(mse)
                valid_equations.append(f_to_eval)
                all_constants.append(constants)
        custom_log['unique_valid_formulas_sampled_percentage'] = (self.params.n_formulas_to_sample - n_invalid) / \
                                                                 self.params.n_formulas_to_sample
        custom_log['unique_formulas_sampled_percentage'] = n_all / self.params.n_formulas_to_sample
        custom_log['unique_valid_to_all_unique'] = (n_all - n_invalid) / n_all
        custom_log['predicate_ok_valid_to_all_valid_unique'] = (n_all - n_invalid - n_false_predicate) / (n_all - n_invalid)
        custom_log['optimize_ok_to_all_valid_unique'] = (n_all - n_invalid - n_false_predicate - n_optimize_failed) / (
                    n_all - n_invalid - n_false_predicate)
        custom_log['domain_ok_to_all_valid_unique'] = (n_all - n_invalid - n_false_predicate - n_optimize_failed -
                                                       n_invalid_domain) / (n_all - n_invalid - n_false_predicate -
                                                                            n_optimize_failed)

        self.stats.save_best_samples(sampled_mses=valid_mses, sampled_formulas=valid_formulas)

        self.stats.write_last_n_to_file(self.params.retrain_file)

        train_batches, _ = rs_train.build_ordered_batches(self.params.retrain_file, solver=self)
        if train_batches is None:
            return None, None, None
        if not self.params.no_retrain:
            train_losses, valid_losses = rs_train.run_epoch(self.model, self.optimizer, train_batches, train_batches,
                                                         kl_coef=self.params.kl_coef)
            tr_loss, tr_rec_loss, tr_kl = train_losses
            v_loss, v_rec_loss, v_kl = valid_losses
            custom_log['retrain_train_loss'] = tr_loss
            custom_log['retrain_train_rec_loss'] = tr_rec_loss
            custom_log['retrain_train_kl_loss'] = tr_kl

            custom_log['retrain_val_loss'] = v_loss
            custom_log['retrain_val_rec_loss'] = v_rec_loss
            custom_log['retrain_val_kl_loss'] = v_kl

        # # TODO(julia) add active learning
        # if self.params.active_learning and epoch % self.params.active_learning_epochs == 0:
        #     next_point = active_learning.pick_next_point(solver=self, custom_log=custom_log,
        #                                                  valid_mses=valid_mses, valid_equations=valid_equations)
        #     self._add_next_point(next_point)
        #     custom_log['next_point_value'] = next_point

        return valid_equations, all_constants, custom_log

    def _get_condition(self, n):
        cond_x = np.repeat(self.xs.reshape(1, -1, self.params.model_params['x_dim']), n, axis=0)
        cond_y = np.repeat(self.ys.reshape(1, -1, 1), n, axis=0)
        return cond_x, cond_y

    def _add_next_point(self, next_point):
        self.xs = np.append(self.xs, next_point).reshape(-1, self.params.model_params['x_dim'])
        self.ys = np.append(self.ys, self.params.true_formula.func(np.array(next_point).reshape(-1, 1)))


class FormulaStatisticsLastN:
    def __init__(self, use_n_last_steps, percentile, simplification):
        self.reconstructed_formulas = []
        self.last_n_best_formulas = []
        self.last_n_best_mses = []
        self.last_n_best_sizes = deque([0] * use_n_last_steps, maxlen=use_n_last_steps)
        self.percentile = percentile
        self.all_best_formulas = []
        self.all_best_mses = []
        self.all_best_per_complexity = {}
        self.simplification = simplification

    def clear_the_oldest_step(self):
        s = self.last_n_best_sizes.popleft()
        self.last_n_best_formulas = self.last_n_best_formulas[s:]
        self.last_n_best_mses = self.last_n_best_mses[s:]

    def save_best_samples(self, sampled_mses, sampled_formulas):
        mse_threshold = np.nanpercentile(sampled_mses + self.last_n_best_mses, self.percentile)
        epoch_best_mses = [x for x in sampled_mses if x < mse_threshold]
        epoch_best_formulas = []
        for i in range(len(sampled_formulas)):
            if sampled_mses[i] < mse_threshold:
                f_string = sampled_formulas[i]
                f_equation = rs_equation.Equation(f_string.split())
                if self.simplification:
                    try:
                        sympy_expr = f_equation.sympy_expr().simplify()
                        simple_tokens = rs_equation.Equation.sympy_to_sting(sympy_expr)
                        simple_f = rs_equation.Equation(simple_tokens)
                        simple_string = " ".join(simple_f._prefix_list)
                        if not simple_f.check_validity()[0] or simple_f.complexity >= f_equation.complexity or \
                                rs_operators.CONST_SYMBOL in simple_string:
                            epoch_best_formulas.append(f_string)
                        else:
                            epoch_best_formulas.append(simple_string)
                    except Exception as e:
                        epoch_best_formulas.append(f_string)
                else:
                    epoch_best_formulas.append(f_string)
        assert len(epoch_best_mses) == len(epoch_best_formulas)

        self.last_n_best_sizes.append(len(epoch_best_formulas))
        self.last_n_best_mses += epoch_best_mses
        self.last_n_best_formulas += epoch_best_formulas

        self.all_best_formulas += epoch_best_formulas
        self.all_best_mses += epoch_best_mses

        for error, formula in zip(sampled_mses, sampled_formulas):
            eq = rs_equation.Equation(formula.split())
            if eq.complexity not in self.all_best_per_complexity:
                self.all_best_per_complexity[eq.complexity] = (formula, error)
            else:
                _, current_error = self.all_best_per_complexity[eq.complexity]
                if error < current_error:
                    self.all_best_per_complexity[eq.complexity] = (formula, error)


    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.last_n_best_formulas))


class FormulaStatisticsQueue:
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.formulas = []
        self.mses = []

    def clear_the_oldest_step(self):
        pass

    def save_best_samples(self, sampled_mses, sampled_formulas):

        all_mses = self.mses + sampled_mses
        all_formulas = self.formulas + sampled_formulas

        sorted_pairs = sorted(zip(all_mses, all_formulas), key=lambda x: x[0])
        used = set()
        unique_pairs = [x for x in sorted_pairs if x[1] not in used and (used.add(x[1]) or True)][:self.queue_size]
        random.shuffle(unique_pairs)

        self.mses = [x[0] for x in unique_pairs]
        self.formulas = [x[1] for x in unique_pairs]

    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.formulas))
