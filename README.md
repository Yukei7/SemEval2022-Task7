# SemEval2022-Task7

## Requirements

Please use the following command to install the required pacakges.

```bash
$ pip install -r requirements.txt
```

## How to Run

For the baseline model, first update the configuration file `baseline/conf.json` to specify the type of Task (ranking/classification) and the file paths.

Then run the following commands to get the baseline results:

```bash
$ cd baseline
$ python main.py
```

For our methods, please update the configuration file `code/config.json` to modify the file paths and training parameters.

Then run the following commands to start training:

```bash
$ cd code
$ python main.py
```

## Reference

> [SemEval2021-Task7 Repo](https://github.com/aishgupta/Quantifying-Humor-Offensiveness) We learn and build our solution in reference to this Repo.
> 
> The starter baseline is from this [Repo](https://github.com/acidAnn/semeval2022_task7_starter_kit)

