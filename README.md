# Rush NNUE

A collection of scripts and utilities to manipulate chess data and train a [NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Networks), especially designed and tailored for the [Rush chess engine](https://github.com/L-Benjamin/rush).

## Thanks

Many thanks to the [Lichess](https://lichess.org/) community for providing free access to their impressive database.

## Usage

I assume you are running linux in the rest of the instructions, but running the scripts on another OS might not involve too much work.

### Dependencies

- You will need the command line tool `bzip2` to extract the pgn archive.

- Python dependencies are listed in the Pipfile. I recommend using [pipenv](https://pipenv.pypa.io/en/latest/) to install them. With pipenv installed, simply do:
```bash
pipenv install
```

You can then prepare the dataset and start training your nets.

### Preparing the dataset

Follow the below steps to prepare a dataset for training your very own NNUE. Please read all the instructions before executing the scripts.
1. Fetch a PGN archive on [lichess.org open database](https://database.lichess.org/). DO NOT EXTRACT IT! In the following example, I am using the July 2021 archive. I can't guarantee anything if you are using data from another source.

2. (If using pipenv) Enter the virtual python environnement with:
```bash
pipenv shell
```

3. Execute the script `scripts/make-dataset.sh` like so:
```bash
make-dataset.sh MYARCHIVE.pgn.bz2
```
You can specifiy the size and count of batches, as well as the output directory, do `make-dataset.sh --help` to get the proper usage of the script.

Your dataset is now ready for use in training!

### Training

TODO

# TODO

+ Script to train a nnue.
+ Actually train a nnue.