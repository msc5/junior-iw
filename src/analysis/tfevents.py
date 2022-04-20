from collections import defaultdict, namedtuple
from typing import List
import tensorflow as tf

import argparse

# https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary

TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])


def extract_images_from_event(event_filename: str, image_tags: List[str]):
    topic_counter = defaultdict(lambda: 0)
    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for v in event.summary.value:
            if v.tag in image_tags:
                if v.HasField('tensor'):  # event for images using tensor field
                    s = v.tensor.string_val[2]  # first elements are W and H
                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()
                    topic_counter[v.tag] += 1
                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)
                    yield tbi


parser = argparse.ArgumentParser(
    prog='tfevents',
    description='Extract images from events.out file'
)
parser.add_argument('--path', help='Path to events.out file', nargs='+')
parser.add_argument('--tag', help='Image tags', nargs='+')

if __name__ == "__main__":

    from rich import print

    opts = vars(parser.parse_args())
    print(opts)

    data = extract_images_from_event(opts['path'], opts['tag'])
    print(data, list(data))
