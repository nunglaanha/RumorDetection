# BERTweet Experiment

This directory keeps the BERTweet experiment separate from the `src/` BERT baseline.

## Setup

```bash
pip install optuna emoji
```

## Train

```bash
python -m Bertweet.train --device cuda
```

The model is saved to `Bertweet/models/bertweet_rumor_classifier`.

## Tune

Single process:

```bash
python -m Bertweet.tune --device cuda --trials 30 --tune-epochs 3
```

Four GPUs, one worker per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m Bertweet.tune --device cuda --trials 20 --tune-epochs 3 --storage sqlite:///Bertweet/results/optuna.db
CUDA_VISIBLE_DEVICES=1 python -m Bertweet.tune --device cuda --trials 20 --tune-epochs 3 --storage sqlite:///Bertweet/results/optuna.db
CUDA_VISIBLE_DEVICES=2 python -m Bertweet.tune --device cuda --trials 20 --tune-epochs 3 --storage sqlite:///Bertweet/results/optuna.db
CUDA_VISIBLE_DEVICES=3 python -m Bertweet.tune --device cuda --trials 20 --tune-epochs 3 --storage sqlite:///Bertweet/results/optuna.db
```

Progress is saved after every trial:

- `Bertweet/results/optuna_best_params.json`
- `Bertweet/results/optuna_trials.csv`
