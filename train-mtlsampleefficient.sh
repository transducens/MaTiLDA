#!/bin/bash

if [ "$GPU_ID" == "" ]; then
  GPU_ID=0
fi
export GPU_ID

export MTLDA_MOSES
export MTLDA_MGIZAPP
export MTLDA_BIWORDS

if [ $# -lt 8 ]
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

### Supported tasks are:
# rev
# src
# wrdp
# swap
# replace
# mono
###
noise=$8

timesProcessed=$9

#alpha
bpeOperationsAux=${10}

#additional options
additionalOptions=""
if [ $# -gt 10 ]
then
  additionalOptions="${11}"
fi

maxLegthAfterBpe=100


if [ $noise == "rev" -o $noise == "src" ]; then
  bpeOperationsAux="none"
fi

set -euo pipefail

source train-steps-fairseq-transformer-base-efficient.sh

#########################################

backtranslationTag=""
removeFirstTokenTask=""
if [[ $additionalOptions == *",taggedbacktranslation,"* ]]; then
  backtranslationTag="◸ "
  removeFirstTokenTask="remove"
else
   # If we want to ignore backtranslated sentences for MTL-DA, we still include the tag
   # in the training corpus, but we ignore it during training with fairseq
   if [[ $additionalOptions == *",nodabacktranslation,"* ]]; then
      removeFirstTokenTask="remove"
   fi
fi


#Avoid all steps if preprocessing has alredy finished and 
#training has been interrupted for some reason

if [ ! -f "$permanentDir/corpus/trainFinal.clean-bpe.$lang1" -a ! -f "$permanentDir/corpus/trainFinal.clean-bpe.$lang1.xz" ]; then

if [ ! -f "$permanentDir/corpus/train.clean-bpe.$lang1" -a ! -f "$permanentDir/corpus/train.clean-bpe.$lang2" ]; then

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

apply_bpe dev tc $lang1
apply_bpe dev tc $lang2

apply_bpe test tc $lang1
apply_bpe test tc $lang2

clean_corpus train bpe clean-bpe

rm $permanentDir/corpus/train.bpe.$lang1 $permanentDir/corpus/train.bpe.$lang2

# Corpora must be synchronous
# 1. Process base corpus
# 2. For each auxiliary task
#    2.1 De-bpe
#    2.2 Augment
#    2.3 Bpe.
#    2.4 DO NOT FILTER

# l. 74
fi

### DE-BPE ORIGINAL TRAINING CORPUS ###
for L in $lang1 $lang2 ; do
  cat $permanentDir/corpus/train.clean-bpe.$L | sed -r 's/(@@ )|(@@ ?$)//g' >  $permanentDir/corpus/train.clean-bpe-debpe.$L
done

prepare_dev_test_sets


for NUM_PROCESSING in $(seq $timesProcessed) ; do

  NUMAUX=2
  MAKE_DATA_INPUT="train"
  NOISES="base"

  NUM_NOISE=0

  for localnoise in $(echo "$noise" | tr '+' '\n') ; do

    NUM_NOISE=$(expr $NUM_NOISE + 1)

    if [[ "$bpeOperationsAux" == *"+"*   ]]; then
      alpha=$( echo "$bpeOperationsAux" | tr '+' '\n' | head -n $NUM_NOISE | tail -n 1 )
    else
      alpha="$bpeOperationsAux"
    fi

    #If the task needs multiple processigs or we are in the first processing, apply steps
    if [  "$NUM_PROCESSING" == "1" -o "$localnoise" == "wrdp" -o "$localnoise" == "wrdp2" -o "$localnoise" == "swap" -o "$localnoise" == "replace" ]; then

      if [ "$localnoise" == "wrdp2" -o "$localnoise" == "src" ]; then
        #output: train${NUMAUX}p$NUM_PROCESSING.clean-bpe.$L
        add_task $localnoise $alpha train clean-bpe train$NUMAUX $NUM_PROCESSING "$removeFirstTokenTask"

      else
        #output: train$NUMAUX.clean-bpe-debpe.$L
        add_task $localnoise $alpha train clean-bpe-debpe train$NUMAUX $NUM_PROCESSING "$removeFirstTokenTask"

        #output: train$NUMAUX.bpe.$L
        for L in $lang1 $lang2 ; do
          apply_bpe train${NUMAUX}p$NUM_PROCESSING clean-bpe-debpe $L
        done

        for L in $lang1 $lang2 ; do
          rm $permanentDir/corpus/train${NUMAUX}p$NUM_PROCESSING.clean-bpe-debpe.$L
        done

        #Just symlink
        for L in $lang1 $lang2 ; do
            ln -s train${NUMAUX}p$NUM_PROCESSING.bpe.$L $permanentDir/corpus/train${NUMAUX}p$NUM_PROCESSING.clean-bpe.$L
        done

      fi

      if [ "$localnoise" == "bpe" ] || [ "$localnoise" == "rev2" ]
      then
         #apply_bpe train$NUMAUX$NUM_PROCESSING tc $lang2
         echo "Not supported auxiliary task"
         exit 1
      fi

      
    else
      #Symlink to NUM_PROCESSING = 1
      for L in $lang1 $lang2 ; do
          ln -s train${NUMAUX}p1.bpe.$L $permanentDir/corpus/train${NUMAUX}p$NUM_PROCESSING.clean-bpe.$L
      done
    fi

    MAKE_DATA_INPUT="$MAKE_DATA_INPUT train$NUMAUX"
    NOISES="$NOISES $localnoise"

    NUMAUX=$(expr $NUMAUX + 1)

    done

  make_data_for_training_sample_efficient "$MAKE_DATA_INPUT" "$NOISES" $NUM_PROCESSING

done

#l 72
fi

if [ ! -f "$permanentDir/model/checkpoints/train.checkpoint_best.pt" ]; then

NOISES="base"
for localnoise in $(echo "$noise" | tr '+' '\n') ; do
	NOISES="$NOISES $localnoise"
done

additionalTrainParams="--write-tensorboard $permanentDir/model/tensorboard"
if [[ $additionalOptions == *",multibackward,"* ]]; then
  additionalTrainParams="$additionalTrainParams --multiple-backward"
fi

if [[ $additionalOptions == *",loggradients,"* ]]; then
  additionalTrainParams="$additionalTrainParams --log-gradients"
fi

if [[ $additionalOptions == *",zerogradients,"* ]]; then
  additionalTrainParams="$additionalTrainParams --zero-negative-gradients"
fi

if [[ $additionalOptions == *",surgerygradients,"* ]]; then
  additionalTrainParams="$additionalTrainParams --surgery-negative-gradients"
fi

if [[ $additionalOptions == *",vaccinegradients,"* ]]; then
	  additionalTrainParams="$additionalTrainParams --vaccine-gradients"
fi

if [[ $additionalOptions == *",exp3,"* ]]; then
	  additionalTrainParams="$additionalTrainParams --exp3 --write-tensorboard $permanentDir/model/tensorboard"
fi

if [[ $additionalOptions == *",pgnorm,"* ]]; then
	  additionalTrainParams="$additionalTrainParams --exp3-reward pgnorm"
fi

if [[ $additionalOptions == *",dummmyreward,"* ]]; then
	  additionalTrainParams="$additionalTrainParams --exp3-reward dummy"
fi

if [[ $additionalOptions == *",exprate:"* ]]; then
    EXPRATE=$(echo "$additionalOptions" | tr ',' '\n' | grep '^exprate' | cut -f 2 -d ':')
	  additionalTrainParams="$additionalTrainParams --exp3-exploration-rate $EXPRATE"
fi

if [[ $additionalOptions == *",maxupdate:"* ]]; then
    MAXUPS=$(echo "$additionalOptions" | tr ',' '\n' | grep '^maxupdate' | cut -f 2 -d ':')
	  additionalTrainParams="$additionalTrainParams --max-update $MAXUPS"
fi

if [[ $additionalOptions == *",bigcorpus,"* ]]; then
  additionalTrainParams="$additionalTrainParams --save-interval-updates 5000 --patience 10"
fi

if [[ $additionalOptions == *",splitbatch,"* ]]; then
  additionalTrainParams="$additionalTrainParams --split-batch 2"
fi

if [[ $additionalOptions == *",nodabacktranslation,"* ]]; then
  additionalTrainParams="$additionalTrainParams --aux-task-ignore-gradients-sentences-starting ◹"

  if [[ $additionalOptions != *",taggedbacktranslation,"* ]]; then
    additionalTrainParams="$additionalTrainParams --remove-backtranslation-mark"
  fi
fi

BASEWEIGHT=""
if [[ $additionalOptions == *",basehalfweight,"* ]]; then
  BASEWEIGHT="0.5"
fi


#NOISES format: "base rev replace"
train_nmt "$NOISES" "$timesProcessed" "$additionalTrainParams" "$BASEWEIGHT"

#l 222
fi


if [ ! -e "$permanentDir/eval/report-train" ]; then

translate_test train

debpe_detruecase_detok_test train
report train

fi

if [ ! -e "$permanentDir/model/tune.log" ]; then


#Not needed: we can reuse training data
#make_data_for_tuning train


tune_nmt

else
 echo "### Omitting finetuning ####"

fi

if [ -e $permanentDir/model/checkpoints/tune.checkpoint_best.pt ]; then
  translate_test tune
  debpe_detruecase_detok_test tune
  report tune
fi

clean
