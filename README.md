# Replay-Constrained Actor-Critic

This repository is based on the [research lightning framework](https://github.com/jhejna/research-lightning).

## Usage
First, create the environment using the `.yaml` files.

For details instructions on launching jobs, see the research lightning repo. Below are basic instructions for training. To run a job, simply run the following command:

```
python train.py --config configs/<config file> --path <path to save logs under ./logs>
```

The different parameter configurations for running the experiments can be found in the `.json` files. Switching between XQL and standard Q-Learning is done by simply changing `alg_kwargs.loss` to `gumbel_resacle` from `mse`.