# Rush NNUE

A collection of scripts and utilities to manipulate chess data and train a [NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Networks), especially designed and tailored for the [Rush chess engine](https://github.com/L-Benjamin/rush).

## Thanks

Many thanks to the [Lichess](https://lichess.org/) community for providing free access to their impressive database.

## Usage

Install the dependencies listed in the Pipfile with the utility of your choice. I recommend [pipenv](https://pipenv.pypa.io/en/latest/).

If you are using pipenv, simply do:
```bash
pipenv install
```

Once finished, do the following command to enter the virtual environnement:
```bash
pipenv shell
```

You can then follow the below steps to prepare the dataset and start training your nets.

### Dataset

Follow the below steps to prepare a dataset for training your very own NNUE:
1. Fetch a PGN archive on [lichess.org open database](https://database.lichess.org/). In the following example, I am using the July 2021 archive.
2. Execute the `src/make-dataset.py` script with a PGN text redirected to stdin and the output redirected to stdout (you will need to extract the archive):
```bash
python src/make-dataset.py < data/lichess_db_standard_rated_2021-07.pgn > data/training_data.txt
```
If you do not wish to extract the entire archive to save disk space, simply pipe the output of `bzip2` to the script, like so:
```bash
bzip2 -dc data/lichess_db_standard_rated_2021-07.pgn.bz2 | python src/make-dataset.py > data/training_data.txt
```
I recommend using this second method.
3. Shuffle the lines of the dataset with a tool like `shuf`:
```bash
shuf data/training_data.txt > data/training_data.txt
```

Your dataset is now ready for use in training!

### Training

TODO

# TODO

+ Script to train a nnue.
+ Actually train a nnue.