#!/bin/bash

MYFULLPATH="$(readlink -f $0)"
CURDIR="$(dirname $MYFULLPATH)"

GPU_ID=${GPU_ID-0}

set -euo pipefail

# Variables to be set in the file
# lang1=$1
# lang2=$2
# permanentDir=$3
# bpeOperations=$4
# trainCorpus=$5
# devCorpus=$6
# testCorpus=$7
# noise=$8
# bpOperationsAux=$9

gpuId=0
temp=/tmp
trainArgs="--arch transformer_wmt_en_de --share-all-embeddings  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --weight-decay 0  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-7 --lr 0.0007 --min-lr 1e-9  --save-interval-updates 1000  --patience 6 --no-progress-bar --max-tokens 4000 --eval-bleu --eval-tokenized-bleu --eval-bleu-args '{\"beam\":5,\"max_len_a\":1.2,\"max_len_b\":10}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-best-checkpoints 1 --keep-interval-updates 1 --no-epoch-checkpoints"
##Descomentar per a executar WMT
#trainArgs="--arch transformer_wmt_en_de --share-all-embeddings  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --weight-decay 0  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-7 --lr 0.0007 --min-lr 1e-9  --save-interval-updates 5000  --patience 6 --no-progress-bar --max-tokens 4000 --eval-bleu --eval-tokenized-bleu --eval-bleu-args '{\"beam\":5,\"max_len_a\":1.2,\"max_len_b\":10}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-best-checkpoints 1 --keep-interval-updates 1 --no-epoch-checkpoints"

moses_scripts=$CURDIR/submodules/moses-scripts/scripts/

nomalizer=$moses_scripts/tokenizer/normalize-punctuation.perl
tokenizer=$moses_scripts/tokenizer/tokenizer.perl
detokenizer=$moses_scripts/tokenizer/detokenizer.perl
clean_corpus=$moses_scripts/training/clean-corpus-n.perl
train_truecaser=$moses_scripts/recaser/train-truecaser.perl
truecaser=$moses_scripts/recaser/truecase.perl
detruecaser=$moses_scripts/recaser/detruecase.perl

apply_noise="python $CURDIR/tools/apply-noise.py"
apply_bil_noise="python $CURDIR/tools/apply-bilingual-noise.py"

prepare_data () {

  prefix=$1  # Prefix to corpus
  tag=$2 #train / dev / test
  prepend_mark="$3"

  echo "prepare_data $prefix $tag ######################"

  if [ ! -e $prefix.$lang1 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang1 does not exist"
    exit 1
  fi

    if [ ! -e $prefix.$lang2 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang2 does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/corpus
  cat $prefix.$lang1 | sed "s:^:$prepend_mark:" > $permanentDir/corpus/$tag.$lang1
  cat $prefix.$lang2 > $permanentDir/corpus/$tag.$lang2
}


prepare_backtranslated_data () {

  prefix=$1  # Prefix to corpus
  tag=$2 #train / dev / test
  slbacktrans=$3
  tlbacktrans=$4

  echo "prepare_data $prefix $tag ######################"

  if [ ! -e $prefix.$lang1 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang1 does not exist"
    exit 1
  fi

    if [ ! -e $prefix.$lang2 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang2 does not exist"
    exit 1
  fi

  if [ ! -e $slbacktrans ]
  then
    echo "prepare_data: ERROR: File $slbacktrans does not exist"
    exit 1
  fi

  if [ ! -e $tlbacktrans ]
  then
    echo "prepare_data: ERROR: File $tlbacktrans does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/corpus
  cat $prefix.$lang1 $slbacktrans > $permanentDir/corpus/$tag.$lang1
  cat $prefix.$lang2 $tlbacktrans > $permanentDir/corpus/$tag.$lang2
}


tokenize () {
  prefix=$1
  lang=$2

  echo "tokenize $prefix $lang ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$lang ]
  then
    echo "tokenize: ERROR: File $permanentDir/corpus/$prefix.$lang does not exist"
    exit 1
  fi

  cat $permanentDir/corpus/$prefix.$lang | $nomalizer -l $lang | $tokenizer -a -no-escape -l $lang > $permanentDir/corpus/$prefix.tok.$lang
}

clean_corpus () {
  prefix=$1 # train / train2
  intag=$2  # tok / bpe
  outtag=$3 # clean / clean-bpe

  echo "clean_corpus $prefix $intag $outtag ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$intag.$lang1 ]
  then
    echo "clean_corpus: ERROR: File $permanentDir/corpus/$prefix.$intag.$lang1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/$prefix.$intag.$lang2 ]
  then
    echo "clean_corpus: ERROR: File $permanentDir/corpus/$prefix.$intag.$lang2 does not exist"
    exit 1
  fi

  paste $permanentDir/corpus/$prefix.$intag.$lang1 $permanentDir/corpus/$prefix.$intag.$lang2 |\
  grep -v "http:" | grep -v "https:" | awk 'BEGIN{FS="[\t]"}{if ($1!=$2) print}' > $permanentDir/corpus/$prefix.$intag.preclean
  cut -f1 $permanentDir/corpus/$prefix.$intag.preclean > $permanentDir/corpus/$prefix.$intag.preclean.$lang1
  cut -f2 $permanentDir/corpus/$prefix.$intag.preclean > $permanentDir/corpus/$prefix.$intag.preclean.$lang2
  $clean_corpus $permanentDir/corpus/$prefix.$intag.preclean $lang1 $lang2  $permanentDir/corpus/$prefix.$outtag 5 100 $permanentDir/corpus/$prefix.$outtag.lines-retained

  rm $permanentDir/corpus/$prefix.$intag.preclean $permanentDir/corpus/$prefix.$intag.preclean.$lang1 $permanentDir/corpus/$prefix.$intag.preclean.$lang2
}

learn_truecaser_train () {
  lang=$1

  echo "learn_truecaser_train $lang ######################"

  if [ ! -e $permanentDir/corpus/train.clean.$lang ]
  then
    echo "learn_truecaser_train: ERROR: File $permanentDir/corpus/train.clean.$lang does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/model/truecaser
  $train_truecaser -corpus $permanentDir/corpus/train.clean.$lang -model $permanentDir/model/truecaser/truecase-model.$lang
}

apply_truecaser () {
  prefix=$1
  intag=$2
  outtag=$3
  lang=$4

  echo "apply_truecaser $prefix $intag $outtag $lang ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$intag.$lang ]
  then
    echo "apply_truecaser: ERROR: File $permanentDir/corpus/$prefix.$intag.$lang does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/model/truecaser/truecase-model.$lang ]
  then
    echo "apply_truecaser: ERROR: File $permanentDir/model/truecaser/truecase-model.$lang does not exist"
    exit 1
  fi

  cat $permanentDir/corpus/$prefix.$intag.$lang | $truecaser -model $permanentDir/model/truecaser/truecase-model.$lang > $permanentDir/corpus/$prefix.$outtag.$lang
}


learn_join_bpe () {
  operations=$1

  if [ $# -eq 2 ]
  then
    ttag=$2
  else
    ttag=""
  fi

  echo "learn_join_bpe $operations $ttag ######################"

  if [ ! -e $permanentDir/corpus/train.tc.$lang1 ]
  then
    echo "learn_join_bpe: ERROR: File $permanentDir/corpus/train.tc.$lang1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/train.tc.$lang2 ]
  then
    echo "learn_join_bpe: ERROR: File $permanentDir/corpus/train.tc.$lang2 does not exist"
    exit 1
  fi

  cat $permanentDir/corpus/train.tc.$lang1 | sed -re 's/\s+/ /g'  > $permanentDir/corpus/train.tc.$lang1.bpeready
  cat $permanentDir/corpus/train.tc.$lang2 | sed -re 's/\s+/ /g'  > $permanentDir/corpus/train.tc.$lang2.bpeready

  subword-nmt learn-joint-bpe-and-vocab --input $permanentDir/corpus/train.tc.$lang1.bpeready $permanentDir/corpus/train.tc.$lang2.bpeready \
              -s $operations -o $permanentDir/model/vocab.$lang1$lang2.bpe$ttag --write-vocabulary \
              $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang1 $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang2
  
  rm $permanentDir/corpus/train.tc.$lang1.bpeready $permanentDir/corpus/train.tc.$lang2.bpeready
}

apply_bpe () {
  prefix=$1
  label=$2
  lang=$3

  if [ $# -eq 4 ]
  then
    ttag=$4
  else
    ttag=""
  fi


  echo "apply_bpe $prefix $label $lang $ttag ######################"

  if [ ! -e $permanentDir/model/vocab.$lang1$lang2.bpe$ttag ]
  then
    echo "apply_bpe: ERROR: File $permanentDir/model/vocab.$lang1$lang2.bpe$ttag does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang1 ]
  then
    echo "apply_bpe: ERROR: File $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang2 ]
  then
    echo "apply_bpe: ERROR: File $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang2 does not exist"
    exit 1
  fi

  cat $permanentDir/corpus/$prefix.$label.$lang |\
  subword-nmt apply-bpe --vocabulary $permanentDir/model/vocab.$lang1$lang2.bpe$ttag.bpevocab.$lang --vocabulary-threshold 1 \
                        -c $permanentDir/model/vocab.$lang1$lang2.bpe$ttag > $permanentDir/corpus/$prefix.bpe.$lang
}

add_task () {
  tasktype=$1
  bpeAux=$2
  prefix=$3
  #Label: processing suffix: tok, tc, etc.
  label=$4
  #Tag = name of the resulting file. train1, train2, etc.
  tag=$5
  numProcessing=$6

  #For tagged backtranslation
  removeFirstToken=$7

  PROC_SUFFIX="p$numProcessing"

  echo "add_task $tasktype $bpeAux $prefix $label $tag ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$label.$lang1 ]
  then
    echo "add_task: ERROR: File $permanentDir/corpus/$prefix.$label.$lang1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/$prefix.$label.$lang2 ]
  then
    echo "add_task: ERROR: File $permanentDir/corpus/$prefix.$label.$lang2 does not exist"
    exit 1
  fi

  cat $permanentDir/corpus/$prefix.$label.$lang1  > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1

  if [ "$tasktype" = "src" ]
  then
    #do not copy backtranslation tag
    if [ "$removeFirstToken" == "remove" ]; then
      cat $permanentDir/corpus/$prefix.$label.$lang1 | cut -f 2- -d ' '  > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang2
    else
      cat $permanentDir/corpus/$prefix.$label.$lang1  > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang2
    fi
  elif [ "$tasktype" = "mono" ]
  then

    ALIGNMENT_INPUT_SUFFIX=""
    #remove backtranslation tag before all the process
    if [ "$removeFirstToken" == "remove" ]; then
      cat $permanentDir/corpus/$prefix.$label.$lang1 | cut -f 2- -d ' ' > $permanentDir/corpus/$prefix.$label.nobackttag.$lang1
      cat $permanentDir/corpus/$prefix.$label.$lang1 | cut -f 1 -d ' ' > $permanentDir/corpus/$prefix.$label.backttag.$lang1
      ln -s $permanentDir/corpus/$prefix.$label.$lang2 $permanentDir/corpus/$prefix.$label.nobackttag.$lang2
      ALIGNMENT_INPUT_SUFFIX=".nobackttag"
    fi


    if [ ! -d "$permanentDir/lexicon" ]; then
      #Call script to build alignments and lexicon
      bash $CURDIR/tools/build_alignments_and_lexicon.sh $lang1 $lang2 $permanentDir $prefix.$label$ALIGNMENT_INPUT_SUFFIX $permanentDir/lexicon "$(nproc)" "$MTLDA_MOSES" "$MTLDA_MGIZAPP"
    fi

    mkdir -p $permanentDir/mono
    bash $CURDIR/tools/gen-from-biwords.sh $lang1 $lang2 $MTLDA_BIWORDS $permanentDir/lexicon/mgizaoutput/giza.$lang2-$lang1/$lang2-$lang1.A3.final.gz $permanentDir/mono

    if [ "$removeFirstToken" == "remove" ]; then
      zcat $permanentDir/mono/$lang1-$lang2.mono.$lang1.gz | paste -d ' ' $permanentDir/corpus/$prefix.$label.backttag.$lang1 - > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1

      #Remove extra files
      rm $permanentDir/corpus/$prefix.$label.backttag.$lang1 $permanentDir/corpus/$prefix.$label.nobackttag.$lang1 $permanentDir/corpus/$prefix.$label.nobackttag.$lang2

    else
      zcat $permanentDir/mono/$lang1-$lang2.mono.$lang1.gz  > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1
    fi
    zcat $permanentDir/mono/$lang1-$lang2.mono.$lang2.gz > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang2
    

  elif [ "$tasktype" = "replace" ] || [ "$tasktype" = "replace_tgt" ]
  then

    ALIGNMENT_INPUT_SUFFIX=""
    #remove backtranslation tag before all the process
    if [ "$removeFirstToken" == "remove" ]; then
      cat $permanentDir/corpus/$prefix.$label.$lang1 | cut -f 2- -d ' ' > $permanentDir/corpus/$prefix.$label.nobackttag.$lang1
      cat $permanentDir/corpus/$prefix.$label.$lang1 | cut -f 1 -d ' ' > $permanentDir/corpus/$prefix.$label.backttag.$lang1
      ln -s $permanentDir/corpus/$prefix.$label.$lang2 $permanentDir/corpus/$prefix.$label.nobackttag.$lang2
      ALIGNMENT_INPUT_SUFFIX=".nobackttag"
    fi

    if [ ! -d "$permanentDir/lexicon" ]; then
      #Call script to build alignments and lexicon
      bash $CURDIR/tools/build_alignments_and_lexicon.sh $lang1 $lang2 $permanentDir $prefix.$label$ALIGNMENT_INPUT_SUFFIX $permanentDir/lexicon "$(nproc)" "$MTLDA_MOSES" "$MTLDA_MGIZAPP"
    fi

    corpora=$permanentDir/lexicon/debpe.train
    alignments=$permanentDir/lexicon/intersection.alignments
    bildic=$permanentDir/lexicon/f2e.lexicon

    paste $corpora.$lang1 $corpora.$lang2 | $apply_bil_noise $tasktype $bpeAux $alignments $bildic > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1-$lang2 2> $permanentDir/corpus/log$PROC_SUFFIX.replace

    if [ "$removeFirstToken" == "remove" ]; then
      cut -f1 $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1-$lang2 | paste -d ' ' $permanentDir/corpus/$prefix.$label.backttag.$lang1 -   > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1

      #Remove extra files
      rm $permanentDir/corpus/$prefix.$label.backttag.$lang1 $permanentDir/corpus/$prefix.$label.nobackttag.$lang1 $permanentDir/corpus/$prefix.$label.nobackttag.$lang2
    else
      cut -f1 $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1-$lang2 > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1
    fi

    cut -f2 $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang1-$lang2 > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang2

  else
    #Everything OK regarding tag because only TL side is mofified
    cat $permanentDir/corpus/$prefix.$label.$lang2 | $apply_noise $tasktype $bpeAux > $permanentDir/corpus/$tag$PROC_SUFFIX.$label.$lang2
  fi
}

__add_to_tag () {
  input=$1
  output=$2
  totag=$3

  if [ ! -e $input ]
  then
    echo "__add_to_tag: ERROR: File $input does not exist"
  fi

  if [ "$noise" == "none" ]  || [ "$noise" == "" ]
  then
    sedcmd="cat -"
  else
    sedcmd="sed -re 's/^/TO_$totag /'"
  fi

  eval "$sedcmd < $input > $output"
}

make_data_for_training () {

  echo "make_data_for_training $@ ######################"

  rm -f $permanentDir/corpus/trainFinal.clean-bpe.$lang1
  rm -f $permanentDir/corpus/trainFinal.clean-bpe.$lang2
  #rm -fr $permanentDir/model/data-bin

  for tag in "$@"
  do
    if [ ! -e $permanentDir/corpus/$tag.clean-bpe.$lang1 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.clean-bpe.$lang1 does not exist"
      exit 1
    fi

    if [ ! -e $permanentDir/corpus/$tag.clean-bpe.$lang2 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.clean-bpe.$lang2 does not exist"
      exit 1
    fi

    if [ "$tag" = "train" ]
    then
      totag=$lang2
    else
      totag=$lang2$tag
    fi

    mv $permanentDir/corpus/$tag.clean-bpe.$lang1 $permanentDir/corpus/$tag.clean-bpe.before-to-tag.$lang1
    __add_to_tag $permanentDir/corpus/$tag.clean-bpe.before-to-tag.$lang1 $permanentDir/corpus/$tag.clean-bpe.$lang1 $totag

    cat $permanentDir/corpus/$tag.clean-bpe.$lang1 >> $permanentDir/corpus/trainFinal.clean-bpe.$lang1
    cat $permanentDir/corpus/$tag.clean-bpe.$lang2 >> $permanentDir/corpus/trainFinal.clean-bpe.$lang2
  done

  fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/trainFinal.clean-bpe \
                     --validpref $permanentDir/corpus/dev.bpe \
                     --destdir $permanentDir/model/data-bin-train --workers 16 --joined-dictionary
}


make_data_for_training_sample_efficient () {

  echo "make_data_for_training_sample_efficient $@ ######################"

  local tags=$1
  local names=$2
  local numProcessing=$3

  #1. If we are in first processing subset:
    # 1.1. add tags, concatenate all the corpora, and create vocabulary
    # 1.2. preprocess main task
    # 1.3. preprocess tasks that do not need to be resampled


  #2. If we are not in first processing subset:
    # 2.1 Symlink files


  #3. Preprocess tasks that need to be resampled



  #1. If we are in first processing subset
  if [  "$numProcessing" == "1" ]; then

    #1.1 concatenate all the corpora and create vocabulary
    rm -f $permanentDir/corpus/trainFinal.clean-bpe.*
    for tag in $tags ; do
      if [ "$tag" = "train" ]
      then
	inputSuffix=""
        totag=$lang2
      else
	inputSuffix="p$numProcessing"
        totag=$lang2$tag
      fi
      __add_to_tag $permanentDir/corpus/$tag$inputSuffix.clean-bpe.$lang1 $permanentDir/corpus/$tag$inputSuffix.clean-bpe-tag.$lang1 $totag
      ln -s $tag$inputSuffix.clean-bpe.$lang2 $permanentDir/corpus/$tag$inputSuffix.clean-bpe-tag.$lang2
      cat $permanentDir/corpus/$tag$inputSuffix.clean-bpe-tag.$lang1 >> $permanentDir/corpus/trainFinal.clean-bpe.$lang1
      cat $permanentDir/corpus/$tag$inputSuffix.clean-bpe.$lang2 >> $permanentDir/corpus/trainFinal.clean-bpe.$lang2
    done
    echo "make_data_for_training_sample_efficient $@ : preprocessing vocabulary ######################"
    python $CURDIR/tools/preprocess-only-dictionary.py -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/trainFinal.clean-bpe --validpref $permanentDir/corpus/dev.bpe --destdir $permanentDir/model/data-bin-vocabulary --workers 16 --joined-dictionary

    #TODO: puedo borrar trainFinal.clean-bpe?

    # 1.2. preprocess main task
    mainTag=$(echo "$tags" | cut -f 1 -d ' ')
    echo "make_data_for_training_sample_efficient $@ : preprocessing main tag numProcessing = $numProcessing ######################"
    fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/$mainTag.clean-bpe-tag --validpref $permanentDir/corpus/dev.bpe --destdir $permanentDir/model/data-bin-train-p$numProcessing --workers 16 --joined-dictionary --srcdict $permanentDir/model/data-bin-vocabulary/dict.$lang1.txt

    # 1.3. preprocess tasks that do not need to be resampled
    numTask=1
    for tag in $tags ; do
      #Omit base task
      if [ $numTask -gt 1  ]; then
        task=$(echo "$names" | cut -f $numTask -d ' ')
        #Check whether this task name needs resampling
        if [ "$task" != "wrdp" -a "$task" != "swap" -a "$task" != "replace" ]; then
           # Preprocess and copy to main directory
	   echo "make_data_for_training_sample_efficient $@ : preprocessing1.3 $tag ($task) numProcessing = $numProcessing ######################"
           fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/${tag}p$numProcessing.clean-bpe-tag --destdir $permanentDir/model/data-bin-train-$task-p$numProcessing --workers 16 --joined-dictionary --srcdict $permanentDir/model/data-bin-vocabulary/dict.$lang1.txt
           for l in $lang1 $lang2 ; do
             cp $permanentDir/model/data-bin-train-$task-p$numProcessing/train.$lang1-$lang2.$l.bin $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.bin
             cp $permanentDir/model/data-bin-train-$task-p$numProcessing/train.$lang1-$lang2.$l.idx $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.idx
           done

           #We can remove output directory because we already copied the result
           rm -R $permanentDir/model/data-bin-train-$task-p$numProcessing
        fi

      fi
      numTask=$(expr $numTask + 1)
    done

  fi

  #2. If we are not in first processing subset:
  if [  $numProcessing -gt 1 ]; then
     mkdir $permanentDir/model/data-bin-train-p$numProcessing
     #2.symlink vocabulary
     ln -s ../data-bin-train-p1/dict.$lang1.txt $permanentDir/model/data-bin-train-p$numProcessing/dict.$lang1.txt
     ln -s ../data-bin-train-p1/dict.$lang2.txt $permanentDir/model/data-bin-train-p$numProcessing/dict.$lang2.txt

     #2.1 Symlink main task
     for l in $lang1 $lang2 ; do
       ln -s ../data-bin-train-p1/train.$lang1-$lang2.$l.bin $permanentDir/model/data-bin-train-p$numProcessing/train.$lang1-$lang2.$l.bin
       ln -s ../data-bin-train-p1/train.$lang1-$lang2.$l.idx $permanentDir/model/data-bin-train-p$numProcessing/train.$lang1-$lang2.$l.idx
     done

     #2.2 Symlink auxiliary tasks that do not need to be resampled
     numTask=1
     for tag in $tags ; do
       #Omit base task
       if [ $numTask -gt 1  ]; then
         task=$(echo "$names" | cut -f $numTask -d ' ')
         #Check whether this task name needs resampling
         if [ "$task" != "wrdp" -a "$task" != "swap" -a "$task" != "replace" ]; then
            for l in $lang1 $lang2 ; do
              ln -s ../data-bin-train-p1/train_aux_$task.$lang1-$lang2.$l.bin $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.bin
              ln -s ../data-bin-train-p1/train_aux_$task.$lang1-$lang2.$l.idx $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.idx
            done
         fi

       fi
       numTask=$(expr $numTask + 1)
     done

     #2.3 Symlink dev set
     for l in $lang1 $lang2 ; do
       ln -s ../data-bin-train-p1/valid.$lang1-$lang2.$l.bin $permanentDir/model/data-bin-train-p$numProcessing/valid.$lang1-$lang2.$l.bin
       ln -s ../data-bin-train-p1/valid.$lang1-$lang2.$l.idx $permanentDir/model/data-bin-train-p$numProcessing/valid.$lang1-$lang2.$l.idx
     done

  fi

  #3. Preprocess tasks that need to be resampled
  numTask=1
  for tag in $tags ; do
    #Omit base task
    if [ $numTask -gt 1  ]; then
      task=$(echo "$names" | cut -f $numTask -d ' ')
      #Check whether this task name needs resampling
      if [ "$task" == "wrdp" -o "$task" == "swap" -o "$task" == "replace" ]; then
        if [  $numProcessing -gt 1 ]; then
	         inputSuffix="p$numProcessing"
           totag=$lang2$tag
	         __add_to_tag $permanentDir/corpus/$tag$inputSuffix.clean-bpe.$lang1 $permanentDir/corpus/$tag$inputSuffix.clean-bpe-tag.$lang1 $totag
           ln -s $tag$inputSuffix.clean-bpe.$lang2 $permanentDir/corpus/$tag$inputSuffix.clean-bpe-tag.$lang2
        fi
        #Preprocess and copy
      	echo "make_data_for_training_sample_efficient $@ : preprocessing3 $tag ($task) numProcessing = $numProcessing ######################"
        fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/${tag}p$numProcessing.clean-bpe-tag --destdir $permanentDir/model/data-bin-train-$task-p$numProcessing --workers 16 --joined-dictionary --srcdict $permanentDir/model/data-bin-vocabulary/dict.$lang1.txt
        for l in $lang1 $lang2 ; do
          cp $permanentDir/model/data-bin-train-$task-p$numProcessing/train.$lang1-$lang2.$l.bin $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.bin
          cp $permanentDir/model/data-bin-train-$task-p$numProcessing/train.$lang1-$lang2.$l.idx $permanentDir/model/data-bin-train-p$numProcessing/train_aux_$task.$lang1-$lang2.$l.idx
        done

        #We can remove directory because we already copied
        rm -R $permanentDir/model/data-bin-train-$task-p$numProcessing

      fi

    fi
    numTask=$(expr $numTask + 1)
  done
}

make_data_for_tuning () {

  echo "make_data_for_tuning $@ ######################"

  rm -f $permanentDir/corpus/tuneFinal.clean-bpe.$lang1
  rm -f $permanentDir/corpus/tuneFinal.clean-bpe.$lang2

  if [ ! -d $permanentDir/model/data-bin-train ]
  then
    echo "make_data_for_tuning: ERROR: Folder $permanentDir/model/data-bin-train does not exist"
    exit 1
  fi

  for tag in "$@"
  do
    if [ ! -e $permanentDir/corpus/$tag.clean-bpe.$lang1 ]
    then
      echo "make_data_for_tuning: ERROR: File $permanentDir/corpus/$tag.clean-bpe.$lang1 does not exist"
      exit 1
    fi

    if [ ! -e $permanentDir/corpus/$tag.clean-bpe.$lang2 ]
    then
      echo "make_data_for_tuning: ERROR: File $permanentDir/corpus/$tag.clean-bpe.$lang2 does not exist"
      exit 1
    fi

    if [ "$tag" = "train" ]
    then
      totag=$lang2
    else
      totag=$lang2$tag
    fi

    #mv $permanentDir/corpus/$tag.clean-bpe.$lang1 $permanentDir/corpus/$tag.clean-bpe.before-to-tag.$lang1
    #__add_to_tag $permanentDir/corpus/$tag.clean-bpe.before-to-tag.$lang1 $permanentDir/corpus/$tag.clean-bpe.$lang1 $totag

    cat $permanentDir/corpus/$tag.clean-bpe.$lang1 >> $permanentDir/corpus/tuneFinal.clean-bpe.$lang1
    cat $permanentDir/corpus/$tag.clean-bpe.$lang2 >> $permanentDir/corpus/tuneFinal.clean-bpe.$lang2
  done

  fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/tuneFinal.clean-bpe \
                     --srcdict $permanentDir/model/data-bin-train/dict.$lang1.txt \
                     --validpref $permanentDir/corpus/dev.bpe \
                     --destdir $permanentDir/model/data-bin-tune --workers 16 --joined-dictionary

                     ##--tgtdict $permanentDir/model/data-bin-train/dict.$lang2.txt
}

prepare_dev_test_sets () {

  echo "prepare_dev_test_sets ######################"

  if [ ! -e $permanentDir/corpus/dev.bpe.$lang1 ]
  then
    echo "prepare_dev_test_sets: ERROR: File $permanentDir/corpus/dev.bpe.$lang1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.bpe.$lang1 ]
  then
    echo "prepare_dev_test_sets: ERROR: File $permanentDir/corpus/test.bpe.$lang1 does not exist"
    exit 1
  fi


  mv $permanentDir/corpus/dev.bpe.$lang1 $permanentDir/corpus/dev.bpe.before_to_tag.$lang1
  __add_to_tag $permanentDir/corpus/dev.bpe.before_to_tag.$lang1 $permanentDir/corpus/dev.bpe.$lang1 $lang2

  mv $permanentDir/corpus/test.bpe.$lang1 $permanentDir/corpus/test.bpe.before_to_tag.$lang1
  __add_to_tag $permanentDir/corpus/test.bpe.before_to_tag.$lang1 $permanentDir/corpus/test.bpe.$lang1 $lang2
}

train_nmt () {
  local NOISES="$1"
  local TIMES_PROCESSED="$2"
  local MOREPARAMS="$3"
  local BASEWEIGHT="$4"

  
  WEIGHTS=""
  NUM_TASKS=$(echo "$NOISES" | tr ' ' '\n' | wc -l)
  if  [ "$BASEWEIGHT" == ""  ]; then
    #Uniform weights
    for i in $(seq $NUM_TASKS) ; do
      W=$(python -c "print(1.0/$NUM_TASKS)")
      WEIGHTS="$WEIGHTS $W"
    done
  else
    WEIGHTS="$BASEWEIGHT"
    for i in $(seq 2 $NUM_TASKS) ; do
	    W=$(python -c "print((1.0 - $BASEWEIGHT)/($NUM_TASKS - 1 ))")
      WEIGHTS="$WEIGHTS $W"
    done
  fi
  
  if [ ! -f "$permanentDir/model/checkpoints/train.checkpoint_best.pt" ]; then
    echo "train_nmt ######################"

    if [ ! -d $permanentDir/model/data-bin-train-p1 ]
    then
      echo "train_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-train-p1 does not exist"
      exit 1
    fi

    echo "Training args: $trainArgs"
    echo "See $permanentDir/model/train.log for details"
    echo "Training command: CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $trainArgs --user-dir $CURDIR/fairseq-modules  --seed $RANDOM --save-dir $permanentDir/model/checkpoints --task translation_efficientsample $MOREPARAMS --assume-reduced-batch --tasks "$NOISES" --weights "$WEIGHTS" --max-epochs-efficient $TIMES_PROCESSED $permanentDir/model/data-bin-train-p   &> $permanentDir/model/train.log"

    eval "CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $trainArgs --user-dir $CURDIR/fairseq-modules  --seed $RANDOM --save-dir $permanentDir/model/checkpoints --task translation_efficientsample $MOREPARAMS --assume-reduced-batch --tasks "$NOISES" --weights "$WEIGHTS" --max-epochs-efficient $TIMES_PROCESSED $permanentDir/model/data-bin-train-p   &> $permanentDir/model/train.log"

    mv $permanentDir/model/checkpoints/checkpoint_best.pt $permanentDir/model/checkpoints/train.checkpoint_best.pt
    rm -fr $permanentDir/model/checkpoints/checkpoint*
  else
    echo "train_nmt ###################### omitted"
  fi
}


tune_nmt () {
  echo "tune_nmt ######################"

  if [ ! -d $permanentDir/model/data-bin-train-p1 ]
  then
    echo "tune_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-train-p1 does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/model/checkpoints/train.checkpoint_best.pt ]
  then
    echo "tune_nmt_fairseq: ERROR: File $permanentDir/model/checkpoints/train.checkpoint_best.pt does not exist"
    exit 1
  fi

  echo "Tune args: $trainArgs"
  echo "See $permanentDir/model/tune.log for details"
  
  
  eval "CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $trainArgs --user-dir $CURDIR/fairseq-modules --seed $RANDOM --task translation_efficientsample --assume-reduced-batch --tasks base --weights 1.0 --max-epochs-efficient 1 --save-dir $permanentDir/model/checkpoints $permanentDir/model/data-bin-train-p --reset-dataloader --restore-file $permanentDir/model/checkpoints/train.checkpoint_best.pt &> $permanentDir/model/tune.log"

  if [ -f "$permanentDir/model/checkpoints/checkpoint_best.pt" ]; then
    mv $permanentDir/model/checkpoints/checkpoint_best.pt $permanentDir/model/checkpoints/tune.checkpoint_best.pt
  fi
  rm -fr $permanentDir/model/checkpoints/checkpoint*
}

translate_test () {
  tag=$1
  echo "translate_test $tag ######################"

  if [ ! -e $permanentDir/model/checkpoints/$tag.checkpoint_best.pt ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/model/checkpoints/$tag.checkpoint_best.pt does not exist"
  fi

  if [ ! -e $permanentDir/corpus/test.bpe.$lang1 ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/corpus/test.bpe.$lang1 does not exist"
    exit 1
  fi

  if [ ! -d $permanentDir/model/data-bin-train-p1 ]
  then
    echo "translate_test_fairseq: ERROR: Folder $permanentDir/model/data-bin-train-p1 does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/eval/
   if [ -e $permanentDir/model/checkpoints/$tag.checkpoint_best.pt ]
  then
  if [ ! -s "$permanentDir/eval/test.output-$tag"  ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive --user-dir $CURDIR/fairseq-modules  --input $permanentDir/corpus/test.bpe.$lang1 --path $permanentDir/model/checkpoints/$tag.checkpoint_best.pt \
                                              $permanentDir/model/data-bin-train-p1 | grep '^H-' | cut -f 3 > $permanentDir/eval/test.output-$tag
  fi
    
    #Compute source influence
    CUDA_VISIBLE_DEVICES=$GPU_ID python $CURDIR/tools/contributions-input-perturbation.py --user-dir $CURDIR/fairseq-modules --path $permanentDir/model/checkpoints/$tag.checkpoint_best.pt --task translation_efficientsample --tasks base  --gen-subset valid --n-perturbations 50 --max-tokens 8000  $permanentDir/model/data-bin-train-p > $permanentDir/eval/test.output-$tag.sourcecontrib
  fi
}

translate_mono () {
  tag=$1
  echo "translate_mono $tag ######################"

  if [ ! -e $permanentDir/model/checkpoints/$tag.checkpoint_best.pt ]
  then
    echo "translate_mono_fairseq: ERROR: File $permanentDir/model/checkpoints/$tag.checkpoint_best.pt does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/mono.bpe.$lang1 ]
  then
    echo "translate_mono_fairseq: ERROR: File $permanentDir/corpus/mono.bpe.$lang1 does not exist"
    exit 1
  fi

  if [ ! -d $permanentDir/model/data-bin-$tag ]
  then
    echo "train_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-$tag does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/eval/

  CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive  --input $permanentDir/corpus/mono.bpe.$lang1 --path $permanentDir/model/checkpoints/$tag.checkpoint_best.pt \
                                              $permanentDir/model/data-bin-$tag | grep '^H-' | cut -f 3 > $permanentDir/eval/mono.output-$tag
}

debpe_detruecase_detok_test () {
  tag=$1
  echo "debpe_detruecase_detok_test $tag ######################"

  if [ ! -e $permanentDir/eval/test.output-$tag ]
  then
    echo "debpe_detruecase_detok_test: ERROR: File $permanentDir/eval/test.output-$tag does not exist"
    exit 1
  fi

  cat $permanentDir/eval/test.output-$tag | sed -r 's/(@@ )|(@@ ?$)//g' > $permanentDir/eval/test.output-$tag.debpe
  cat $permanentDir/eval/test.output-$tag.debpe |  $detruecaser > $permanentDir/eval/test.output-$tag.detruecased
  cat $permanentDir/eval/test.output-$tag.detruecased  | $detokenizer -l $lang2 > $permanentDir/eval/test.output-$tag.detokenized
}


debpe_detruecase_detok_mono () {
  tag=$1
  echo "debpe_detruecase_detok_mono $tag ######################"

  if [ ! -e $permanentDir/eval/mono.output-$tag ]
  then
    echo "debpe_detruecase_detok_mono: ERROR: File $permanentDir/eval/mono.output-$tag does not exist"
    exit 1
  fi

  cat $permanentDir/eval/mono.output-$tag | sed -r 's/(@@ )|(@@ ?$)//g' > $permanentDir/eval/mono.output-$tag.debpe
  cat $permanentDir/eval/mono.output-$tag.debpe |  $detruecaser > $permanentDir/eval/mono.output-$tag.detruecased
  cat $permanentDir/eval/mono.output-$tag.detruecased  | $detokenizer -l $lang2 > $permanentDir/eval/mono.output-$tag.detokenized
}

report () {
  tag=$1
  echo "report $tag ######################"

  if [ ! -e $permanentDir/eval/test.output-$tag.detokenized ]
  then
    echo "report: ERROR: File $permanentDir/eval/test.output-$tag.detokenized does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.$lang2 ]
  then
    echo "report: ERROR: File $permanentDir/corpus/test.$lang2 does not exist"
    exit 1
  fi

  cat $permanentDir/eval/test.output-$tag.detokenized | sacrebleu $permanentDir/corpus/test.$lang2 --width 3 -l $lang1-$lang2 --metrics bleu chrf  > $permanentDir/eval/report-$tag
}

clean () {
  echo "clean ######################"

  rm -f $permanentDir/corpus/train.*
  rm -f $permanentDir/corpus/train[0-9]*
  rm -f $permanentDir/corpus/dev.$lang1 $permanentDir/corpus/dev.$lang2 $permanentDir/corpus/dev.tok.* $permanentDir/corpus/dev.tc.*
  rm -f $permanentDir/corpus/test.$lang1 $permanentDir/corpus/test.$lang2 $permanentDir/corpus/test.tok.* $permanentDir/corpus/test.tc.*
  rm -f $permanentDir/corpus/*.before_to_tag.*
  xz $permanentDir/corpus/trainFinal.*
  if [ -e $permanentDir/corpus/tuneFinal.clean-bpe.$lang1 ]
  then
    xz $permanentDir/corpus/tuneFinal.*
  fi
  cd $permanentDir/model

  #tar cvfJ data-bin-train-p1.tar.xz data-bin-train-p1
  rm data-bin-train-p*/*.bin data-bin-train-p*/*.idx
  cd -
}
