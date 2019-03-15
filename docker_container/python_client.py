'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

import time
from argparse import ArgumentParser

import grpc
import numpy as np
import cv2
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the image')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='0.0.0.0:9000',
                        help='prediction service host:port')
    parser.add_argument("-i", "--image",
                        dest="image",
                        default='',
                        help="path to image in JPEG format", )
    parser.add_argument('-p', '--image_path',
                        dest='image_path',
                        default='/home/baidu/Pictures/dota2_1.jpg',
                        help='path to images folder', )
    parser.add_argument('-b', '--batch_mode',
                        dest='batch_mode',
                        default='true',
                        help='send image as batch or one-by-one')
    args = parser.parse_args()

    return args.server, args.image, args.image_path, args.batch_mode == 'true'


def main():
    """

    :return:
    """
    server, image, image_path, batch_mode = parse_args()

    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, np.float32)

    image_list = []

    for i in range(128):
        image_list.append(image)

    image_list = np.array(image_list, dtype=np.float32)

    start = time.time()

    if batch_mode:
        print('In batch mode')
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'nsfw'
        request.model_spec.signature_name = 'classify_result'

        request.inputs['input_tensor'].CopyFrom(make_tensor_proto(
            image_list, shape=[128, 224, 224, 3]))

        try:
            result = stub.Predict(request, 60.0)
        except Exception as err:
            print(err)
            return
        print(result)
    else:
        return

    end = time.time()
    time_diff = end - start
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    main()
