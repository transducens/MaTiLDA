# MaTiLDA

This repository contains the code needed to run _multi-task learning data augmentation_ (MaTiLDA), a method for neural machine translation data augmentation presented in the paper:

V. M. Sánchez-Cartagena, M. Esplà-Gomis, J. A. Pérez-Ortiz and F. Sánchez-Martínez, "Non-Fluent Synthetic Target-Language Data Improve Neural Machine Translation," in _IEEE Transactions on Pattern Analysis and Machine Intelligence_, doi: 10.1109/TPAMI.2023.3333949.

## Set up environment and install software dependencies

MaTiLDA is distributed as a set of modules for fairseq. It has been tested with fairseq 0.10.2 and Python 3.7. In order to set up the environment, please follow these steps.

Create a Python virtualenv and activate it:

```
virtualenv -p python3.7 ~/envs/matilda
source ~/envs/matilda/bin/activate
```

Clone and init submodules:
``` 
git clone https://github.com/transducens/MaTiLDA.git
cd MaTiLDA
git submodule update --init --recursive
```

Install dependencies:
```
pip install -r requirements.txt
```

## Compile MGIZA++

If you are going to add the "replace" or "mono" auxiliary tasks, you will need to install MGIZA++ as follows. You can skip this section if you are not going to produce synthetic data with these auxiliary tasks. Execute these commands inside the MaTiLDA working copy diectory:

```
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
mkdir build && cd build
cmake ..
make
ln -s $PWD/../scripts/merge_alignment.py $PWD/bin/merge_alignment.py
cd ../../..
```

Once finished, export the Bash environment variable `MTLDA_MGIZAPP` with the path to the MGIZA++ installation directory. The training scripts make use of this environment variable.

```
export MTLDA_MGIZAPP=$PWD/mgiza/mgizapp/build/bin/
```

## Download data

You can download part of the corpora we used in our experiments (low-resource scenario for en-de, en-he and en-vi) as follows:

```
wget http://www.dlsi.ua.es/~vmsanchez/emnlp2021-data.tar.gz
tar xvzf emnlp2021-data.tar.gz
```

## Train baseline systems

In order to train a baseline system, run the script shown below, where the Bash variables have the following meaning:
* $L1 and $L2: source and target languages codes. Use `en` for English, `de` for German, `he` for Hebrew and `vi` for Vietnamese.
* $PAIR: language pair. We always consider English as the first language of the pair, regardless of whether it acts as the source of the target language. Possible values are `en-de`, `en-he`, and `en-vi`.
* $DIR: path to the directory that will be created during the training process and will contain files with the intermediate steps and results.
* $bpe: number of BPE merge operations. We used 10000 in all the experiments reported in the paper.
* $TRAINSET: training data to use. Use `iwslt` to use the data from the downloaded package.

```
./train-baseline.sh $L1 $L2 $DIR $bpe data/$TRAINSET-$PAIR/train data/$TRAINSET-$PAIR/dev data/$TRAINSET-$PAIR/test
```

You can find the resulting BLEU and chrF++ scores in the file `$DIR/eval/report-train`

By default, the GPU 0 as shown by the the `nvidia-smi` command will be used to train the system. If you want to use another GPU, prepend the string `CUDA_VISIBLE_DEVICES=NUM_GPU` to the training command, as in the following example:

```
CUDA_VISIBLE_DEVICES=2 ./train-baseline.sh $L1 $L2 $DIR $bpe data/$TRAINSET-$PAIR/train data/$TRAINSET-$PAIR/dev data/$TRAINSET-$PAIR/test
```


## Train systems with "reverse" or "source" auxiliary tasks

The Bash variables have the same meaning as in the previous section, and we have a new one:
* $AUXTASK: use `rev` for training with the "reverse" auxiliary task and `src` for training with the "source" auxiliary task.

```
./train-mtlsampleefficient.sh $L1 $L2 $DIR $bpe data/$TRAINSET-$PAIR/train data/$TRAINSET-$PAIR/dev data/$TRAINSET-$PAIR/test $AUXTASK
```

You can find the resulting BLEU and chrF++ scores in the file `$DIR/eval/report-tune`. If that file does not exists because BLEU in the development set did not improve during finetuning, scores can be found in the file `$DIR/eval/report-tune`.

## Train systems with "token" or "swap" auxiliary tasks

The "token" and "swap" auxiliary tasks require an alpha parameter that controls the proportion of the sentence which is modified. Moreover, as the random modifications are different in each training epoch, we need to specify the number of expected training epochs in order to precompute the modifications and save training time. To be on the safe side, it is recommended to use 1.5 the number of training epochs of a baseline system without MaTiLDA. This is the meaning of the Bash variables used in the script below:

* $AUXTASK: use `wrdp2` for training with the "token" auxiliary task and `swap` for training with the "swap" auxiliary task.
* $ALPHA: proportion of the tokens in the target sentence that are modified.
* $PRECOMPEPOCHS:  number of expected training epochs.

```
./train-mtlsampleefficient.sh $L1 $L2 $DIR $bpe data/$TRAINSET-$PAIR/train data/$TRAINSET-$PAIR/dev data/$TRAINSET-$PAIR/test $AUXTASK $PRECOMPEPOCHS $ALPHA
```

## Train systems with "replace" auxiliary task

The "replace" task requires word-aligning the training data and extracting a bilingual lexicon from it. In addition to MGIZA++, we will need a working installation of Moses. Please follow the [Moses official installation instructions](http://www.statmt.org/moses/?n=Development.GetStarted). Once installed, export the Bash environment variable MTLDA_MOSES with the path to the Moses root directory, as in the following example:

```
export MTLDA_MOSES=/home/myuser/software/mosesdecoder
```

Once the envitonment variables `MTLDA_MOSES` and `MTLDA_MGIZAPP` have been exported, you can train a system by issuing a command similar to the ones depicted for other auxiliary tasks:

```
./train-mtlsampleefficient.sh $L1 $L2 $DIR $bpe data/$TRAINSET-$PAIR/train data/$TRAINSET-$PAIR/dev data/$TRAINSET-$PAIR/test replace $PRECOMPEPOCHS $ALPHA
```


