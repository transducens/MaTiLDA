#! /bin/bash

lang1=$1
lang2=$2
corpusDir=$3
inputPrefix=$4
permanentDir=$5
njobs=$6
curdir=$(realpath $(dirname $0))

mosesDir=$7
mgizappDir=$8

#Taking pre-processed training data and reverting BPE
mkdir -p $permanentDir
cat $corpusDir/corpus/$inputPrefix.$lang1  | sed -r 's/(@@ )|(@@ ?$)//g' > $permanentDir/debpe.train.$lang1
cat $corpusDir/corpus/$inputPrefix.$lang2  | sed -r 's/(@@ )|(@@ ?$)//g' > $permanentDir/debpe.train.$lang2

#Running Moses word alignment (MGIZA++) to obtain word-alignments and a probabilistic lexicon
$mosesDir/scripts/training/train-model.perl  --alignment intersection --root-dir $permanentDir/mgizaoutput --corpus $permanentDir/debpe.train -e $lang2  -f $lang1 --mgiza --mgiza-cpus=$njobs --parallel --first-step 1 --last-step 4 --external-bin-dir $mgizappDir --sort-compress gzip

#Extracting most-likely entries from lexicon, and building monolingual version of the bilingual lexicon
cat $permanentDir/mgizaoutput/model/lex.e2f > $permanentDir/e2f.lexicon
cat $permanentDir/mgizaoutput/model/lex.f2e > $permanentDir/f2e.lexicon

#Moving the symmetrised alignments to the output directory
mv $permanentDir/mgizaoutput/model/aligned.intersection $permanentDir/intersection.alignments
