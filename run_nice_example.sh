#!/usr/bin/env bash

# this would not work, please download manually
#wget --no-proxy https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip

wget --no-proxy https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz