# REACTOR-torch

**WORK IN PROGRESS:** In this repo we are building the first open-source implementation of the REACTOR[1] architecture.
We are starting from an ACER[1] implementation, adding the Distributional updates and beta-LOO loss.

## ACER
ACER[2] is an Actor-Critic Off-Policy Reinforcement Learning algorithm.
Its main features are:
- Multi-step Experience Replay
- RETRACE[3] low-variance estimator for multi step returns
- Multi-threading
- Efficient Trust Region constraint

## Implementation
This implementation relies on PyTorch and Numpy, and at the moment only supports
Gym environments. The main feature that differs from other PyTorch implementations is that
it fully exploits traces collected in the on-policy steps. Many implementations instead cut
the sampled traces to enforce the same length, and optionally sample a starting point. While
this feature is implemented, the default is to always use the full traces for training.

```python
usage: main.py [-h] --env-name ENV_NAME [--exp-name EXP_NAME]
               [--num-workers NUM_WORKERS] [--t-max T_MAX]
               [--worker-steps WORKER_STEPS] [--batch-size BATCH_SIZE]
               [--replay-ratio REPLAY_RATIO] [--buffer-len BUFFER_LEN]
               [--start-train-at START_TRAIN_AT] [--discount DISCOUNT] [--c C]
               [--trust-region]
               [--trust-region-threshold TRUST_REGION_THRESHOLD] [--tau TAU]
               [--lr LR] [--reward-clip] [--entropy-weight ENTROPY_WEIGHT]
               [--max-grad-norm MAX_GRAD_NORM] [--debug] [--no-lock]
               [--no-lock-update]

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Gym environment full name.
  --exp-name EXP_NAME   Name of the experiment.
  --num-workers NUM_WORKERS
                        Number of workers to be spawned (one process for
                        each). Defaults to 4.
  --t-max T_MAX         Total number of steps to run in the environment.
                        Defaults to 1e6.
  --worker-steps WORKER_STEPS
                        Number of steps that each worker will collect in the
                        environment before performing on and off policy
                        updates. Defaults to 1e3.
  --batch-size BATCH_SIZE
                        Size of trajectory batches sampled in off-policy
                        training. Defaults to 32.
  --replay-ratio REPLAY_RATIO
                        Expected value of off-policy trainings for each on-
                        policy training. Defaults to 4.
  --buffer-len BUFFER_LEN
                        Capacity of the replay buffer measured in
                        trajectories. The total number of transitions is
                        therefore worker steps * buffer len. Defaults to 1e4.
  --start-train-at START_TRAIN_AT
                        How many steps to perform before to start off-policy
                        training. On-policy training is always performed from
                        the beginning. Defaults to 0 (start train when enough
                        samples to take a batch).
  --discount DISCOUNT   Discount factor. Defaults to 0.99.
  --c C                 Retrace importance weight truncation. Defaults to 1.
  --trust-region        Use trust region constraint in policy updates.
                        Defaults to False.
  --trust-region-threshold TRUST_REGION_THRESHOLD
                        Trust region threshold value (delta in Eq. 12 of ACER
                        paper). Defaults to 1.
  --tau TAU             Weight of current average policy when updating it with
                        the new policy. avg_p = tau * avg_p + (1 - tau) p.
                        Defaults to 0.99.
  --lr LR               Learning rate. Defaults to 5e-4.
  --reward-clip         Clip rewards in range [-1, 1]. Defaults to False.
  --entropy-weight ENTROPY_WEIGHT
                        Entropy regularisation weight. Defaults to 1e-4.
  --max-grad-norm MAX_GRAD_NORM
                        Gradient L2 normalisation. Defaults to 40.
  --debug               Run a single thread and without tensorboard. Defaults
                        to False.
  --no-lock             Do not use the Lock mechanism when transferring
                        gradients to the shared model. Potentially allows
                        workers to overwrite each other's work, but may
                        improve speed. Default is to use Lock.
  --no-lock-update      Do not use the Lock mechanism when performing the
                        optimization step on the shared model. Potentially
                        allows worker to perform optimization at the same
                        time, but may improve speed. Default is to use Lock
                        both in gradient transfer and in optimization.
```

# Roadmap
- [x] Trace Experience Replay
- [x] RETRACE estimation
- [x] Efficient Trust region
- [x] Discrete actions  
- [x] Multi-threading
- [x] Tensorboard
- [ ] Continuous actions
- [ ] Full trust region option
- [ ] Load and save model

# References
[1] Gruslys et al.,
[The Reactor: A fast and sample-efficient Actor-Critic agent for Reinforcement Learning](https://arxiv.org/abs/1704.04651)

[2] 2017, Wang et al. 
[Sample Efficient Actor Critic with Experience Replay](https://arxiv.org/pdf/1611.01224.pdf)

[3] 2016, Munos et al.
[Safe and efficient off-policy reinforcement learning](https://arxiv.org/pdf/1606.02647.pdf)
