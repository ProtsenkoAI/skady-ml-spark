from .train_pipeline_scheduler import TrainPipelineScheduler


class SchedulerBuilder:
    """The class is needed because scheduler needs number of steps in epoch to be inited, but
    in train_pipeline we know the number of steps only when receive dataset - when user calls run().
    So we set all params here, and then build scheduler with knowledge of the nsteps per epoch."""
    def __init__(self, *scheduler_args, **scheduler_kwargs):
        self.args = scheduler_args
        self.kwargs = scheduler_kwargs

    def build(self, nsteps_per_epoch):
        return TrainPipelineScheduler(nsteps_per_epoch, *self.args, **self.kwargs)
