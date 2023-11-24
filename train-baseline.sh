#!/bin/bash 

if [ "$GPU_ID" == "" ]; then
  GPU_ID=0
fi

export GPU_ID

set -euo pipefail

if [ $# -lt 7 ]
then
  echo "Wrong number of arguments"
  exit 1
fi

lang1=$1
lang2=$2
permanentDir=$3
bpeOperations=$4

trainCorpus=$5
devCorpus=$6
testCorpus=$7

maxLegthAfterBpe=100
noise="none"

additionalOptions=""
if [ $# -gt 7 ]
then
  additionalOptions="$8"
fi

source train-steps-fairseq-transformer-base-baseline.sh

#########################################

if [ ! -d "$permanentDir/model/data-bin-train" ]
then

backtranslationTag=""
if [[ $additionalOptions == *",taggedbacktranslation,"* ]]; then
  backtranslationTag="◸ "
fi


prepare_data $trainCorpus train ""
prepare_data $testCorpus test "$backtranslationTag"
prepare_data $devCorpus dev "$backtranslationTag"

  tokenize train $lang1
  tokenize train $lang2

  rm $permanentDir/corpus/train.$lang1 $permanentDir/corpus/train.$lang2

  tokenize test $lang1
  tokenize test $lang2

  tokenize dev $lang1
  tokenize dev $lang2

  clean_corpus train tok clean

  learn_truecaser_train $lang1
  learn_truecaser_train $lang2

  apply_truecaser train clean tc $lang1
  apply_truecaser train clean tc $lang2

  apply_truecaser dev tok tc $lang1
  apply_truecaser dev tok tc $lang2

  apply_truecaser test tok tc $lang1
  apply_truecaser test tok tc $lang2

  rm $permanentDir/corpus/train.clean.$lang1 $permanentDir/corpus/train.clean.$lang2


  learn_join_bpe $bpeOperations

  apply_bpe train tc $lang1
  apply_bpe train tc $lang2

  rm $permanentDir/corpus/train.tc.$lang1 $permanentDir/corpus/train.tc.$lang2

  apply_bpe dev tc $lang1
  apply_bpe dev tc $lang2

  apply_bpe test tc $lang1
  apply_bpe test tc $lang2

  clean_corpus train bpe clean-bpe

  rm $permanentDir/corpus/train.bpe.$lang1 $permanentDir/corpus/train.bpe.$lang2
  
  prepare_dev_test_sets

 

  make_data_for_training train
fi

if [ ! -f "$permanentDir/model/checkpoints/train.checkpoint_best.pt" ]
then
  additionalTrainParams=""
  if [[ $additionalOptions == *",bigcorpus,"* ]]; then
    additionalTrainParams="$additionalTrainParams --save-interval-updates 5000 --patience 10"
  fi

  train_nmt "$additionalTrainParams"
fi

translate_test train
debpe_detruecase_detok_test train
report train

clean
