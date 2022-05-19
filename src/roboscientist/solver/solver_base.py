class BaseSolver:
    """
    Base class for formula.
    """
    def __init__(self, logger, *args, **kwargs):
        self._logger = logger
        self._epoch = 0

    def log_metrics(self, reference_dataset, candidate_equations, all_constants, custom_log):
        self._logger.log_metrics(reference_dataset, candidate_equations, all_constants)
        self._logger.commit_metrics(custom_log)

    def solve(self, reference_dataset, epochs: int=100):
        candidate_equations = None
        for epoch in range(epochs):
            self._epoch = epoch
            candidate_equations, all_constants, custom_log = self._training_step(reference_dataset, epoch)
            if candidate_equations is None:
                break
            self.log_metrics(reference_dataset, candidate_equations, all_constants, custom_log)

        return candidate_equations

    def _training_step(self, reference_dataset, epoch):
        raise NotImplementedError('func is not implemented.')
