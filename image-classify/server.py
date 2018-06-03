from flask import Flask, render_template, request
app = Flask(__name__)

import os
import sys
import time
import tensorflow as tf


def file2label(image_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        score_l = map(lambda x: [label_lines[x], predictions[0][x]], top_k)
        if score_l[0][1] > score_l[1][1]:
          return score_l[0][0]
        else:
          return score_l[1][0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    file_path = os.path.join(os.path.dirname(__file__), 'uploads/' + str(time.time()) + '.jpg')
    f.save(file_path)
    return file2label(file_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
