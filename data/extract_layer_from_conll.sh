#!/bin/bash
#title		: extract_layer_from_conll.sh
#description	: This script takes layer name as input, downloads and executes relevant script to extract *_layer from *_conll files present in conll-formatted-ontonotes-5.0 directory.
#author		: sanketvmehta
#date		: 2018-02-09
#version	: 1.0
#usage		: bash extract_layer_from_conll.sh layer_name
#============================================================================

CONLL_FORMAT_ONTO=conll-formatted-ontonotes-5.0
if [ -d $CONLL_FORMAT_ONTO ]; then
   echo "The directory '$CONLL_FORMAT_ONTO' exists."
else
   echo "'$CONLL_FORMAT_ONTO' directory is missing."
fi

if [ "$1" == "" ]; then
  echo "$0: Please provide the layer (parse, name, coreference) which you want to extract from *_conll files."
  echo "usage: $0 layer_name"
  exit 1
fi

CONLL_FORMAT_ONTO_SCRIPTS=conll-formatted-ontonotes-5.0/scripts
CONLL_BASH_FILE_NAME=conll2$1.sh
CONLL_PYTHON_FILE_NAME=conll2$1.py

if [ ! -f $CONLL_FORMAT_ONTO_SCRIPTS/conll2$1.sh ]; then
  echo "Processing scripts are missing! Downloading them from server!"
  
  FILE_NAME=conll-2012-scripts.v3.tar.gz
  if [ -f $FILE_NAME ]; then
     rm $FILE_NAME
  fi
  wget http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
  tar -xvzf conll-2012-scripts.v3.tar.gz -C $CONLL_FORMAT_ONTO
  cp $CONLL_FORMAT_ONTO/conll-2012/v3/scripts/conll2$1* $CONLL_FORMAT_ONTO_SCRIPTS/ 
  rm -r $CONLL_FORMAT_ONTO/conll-2012
 
  FIND="conll2$1.py"
  echo "$FIND"
  REPLACE="python2.7\ conll2$1.py"
  echo "$REPLACE"
  sed -i "s@$FIND@$REPLACE@" $CONLL_FORMAT_ONTO_SCRIPTS/$CONLL_BASH_FILE_NAME 
else
  echo "Processing scripts are present!"
fi

CURR_DIR=$PWD

cd $CONLL_FORMAT_ONTO_SCRIPTS
bash $CONLL_BASH_FILE_NAME -v $CURR_DIR/$CONLL_FORMAT_ONTO
