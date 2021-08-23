# Rush NNUE

A collection of scripts and utilities to manipulate chess data and train a [NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Networks), especially designed and tailored for the [Rush chess engine](https://github.com/L-Benjamin/rush).

## Thanks

Many thanks to the [Lichess](https://lichess.org/) community for providing free access to their impressive database.

## Usage

I assume you are running linux in the rest of the instructions, but running the scripts on another OS might not involve too much work.

### Dependencies

- You will need to have the GNU coreutils installed.

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
You can specifiy the size and count of batches, as well as the output directory, do `make-dataset.sh --help` to get the proper usage of the script. The average disk usage of batches is around 11.89 bytes/position.

4. Execute the script `src/split.py` to parse the data generated into tensors for training. Pass theargument `--help` to get the usage of the script. This will result in two directories, one with your trainig data and one with your testing data. You can expect the generated files to take about 4.5 times more space than the unparsed ones.

Your dataset is now ready for use in training!

### Training

1. (If using pipenv) Enter the virtual python environnement with:
```bash
pipenv shell
```

2. Check if CUDA drivers are properly installed with:
```shell
python src/train.py cuda
```
Those are not required but they will greatly accelerate training and testing.

3. Train your net with the `train` subcommand:
```shell
python src/train.py train --help
```

4. After training for the desired number of epochs, you can use the `test` subcommand to test your network against your testing data:
```shell
python src/train.py test --help
```

### Exporting 

Your nets are stored in pytorch files by default (`.pt`). My scripts allow exporting in JSON, an ubiquitous format, albeit not the most efficient one. A single net in json will result in a ~120MB file. Simply use the `json` subcommand.

### Using

I provide an executable, written in rust, that converts a json net into a format usable by the Rush chess engine. Inference code can be found in it's [repository](https://github.com/L-Benjamin/rush).

## Resources

- [Chess Programming Wiki - NNUE](https://www.chessprogramming.org/NNUE)
- [David Carteau - Cerebrum](https://github.com/david-carteau/cerebrum)
