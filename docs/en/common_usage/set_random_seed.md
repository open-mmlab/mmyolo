# Set the random seed

If you want to set the random seed during training, you can use the following command.

```shell
python ./tools/train.py \
    ${CONFIG} \                               # path of the config file
    --cfg-options randomness.seed=2023 \      # set seed to 2023
    [randomness.diff_rank_seed=True] \        # set different seeds according to global rank
    [randomness.deterministic=True]           # set the deterministic option for CUDNN backend
# [] stands for optional parameters, when actually entering the command line, you do not need to enter []
```

`randomness` has three parameters that can be set, with the following meanings.

- `randomness.seed=2023`, set the random seed to 2023.
- `randomness.diff_rank_seed=True`, set different seeds according to global rank. Defaults to False.
- `randomness.deterministic=True`, set the deterministic option for cuDNN backend, i.e., set `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark` to False. Defaults to False. See https://pytorch.org/docs/stable/notes/randomness.html for more details.
