#!/bin/bash


if [ $# -ne 5 ]
then
  echo "Error: wrong number of parameters"
  exit
fi

L1=$1
L2=$2
BIWORDS=$3
FILE=$4
OUTDIR=$5


echo "Processing alignments ###################"
zcat $FILE | $BIWORDS/giza++A3-to-txt | gzip > $OUTDIR/$L1-$L2.alg.gz

echo "Generating biwords ###################"
$BIWORDS/gen-text-to-compress -a $OUTDIR/$L1-$L2.alg.gz -o -n -z  | gzip > $OUTDIR/$L1-$L2.biwords.gz

echo "Generating mono file for $L1 ###################"
zcat $OUTDIR/$L1-$L2.biwords.gz | cut -d'|' -f1  | awk '{if (length($0)>0) print }'  | tr "\n" " " |  sed 's/^Ɛ//' | sed -re "s/Ɛ/\n/g" | sed -re "s/^\s+//g" | sed -re "s/\s+$//g" | gzip > $OUTDIR/$L1-$L2.mono.$L1.gz
echo "Mono file contains "$(zcat $OUTDIR/$L1-$L2.mono.$L1.gz | wc -l)" lines"

echo "Generating mono file for $L2 ###################"
zcat $OUTDIR/$L1-$L2.biwords.gz | cut -d'|' -f2  | awk '{if (length($0)>0) print }'  | tr "\n" " " |  sed 's/^Ɛ//' | sed -re "s/Ɛ/\n/g" | sed -re "s/^\s+//g" | sed -re "s/\s+$//g" | gzip > $OUTDIR/$L1-$L2.mono.$L2.gz
echo "Mono file contains "$(zcat $OUTDIR/$L1-$L2.mono.$L2.gz | wc -l)" lines"

