#!/bin/bash

SELF_DIR=$(dirname $0)
TARGET_DIR=$SELF_DIR/raw
echo "Downloading into $TARGET_DIR..."

mkdir -p $TARGET_DIR/processed
mkdir -p $TARGET_DIR/raw

META_URL=http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz
wget $META_URL -P $TARGET_DIR/processed
tar -zxvf $TARGET_DIR/processed/kth.tar.gz -C $TARGET_DIR/processed/
rm $TARGET_DIR/processed/kth.tar.gz

DATA_URL=http://www.nada.kth.se/cvap/actions/
DATA_NAMES=(walking jogging running handwaving handclapping boxing)
DATA_URLS=$(for n in ${DATA_NAMES[@]}; do echo ${DATA_URL}${n}.zip; done)

# Download zip files
echo $DATA_URLS | xargs -n 1 -P 8 wget -q -P $TARGET_DIR/raw

for c in walking jogging running handwaving handclapping boxing
do
	mkdir $TARGET_DIR/raw/$c
	unzip $TARGET_DIR/raw/"$c".zip -d $TARGET_DIR/raw/$c
done

python $SELF_DIR/convert.py --data_dir $TARGET_DIR

