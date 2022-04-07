

SELF_DIR=$(dirname $0)
TARGET_DIR=$SELF_DIR/raw
echo "Downloading into $TARGET_DIR..."

mkdir $TARGET_DIR/
URL=http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
wget $URL -P $TARGET_DIR
tar -xvf $TARGET_DIR/bair_robot_pushing_dataset_v0.tar -C $TARGET_DIR

python $SELF_DIR/convert.py --data_dir $TARGET_DIR
