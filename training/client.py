import time
from argparse import ArgumentParser

from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for text classification')

    parser.add_argument('-s', '--server',
                        dest='server',
                        default='172.17.0.2:9000',
                        help='Service host:port')
    parser.add_argument('-t', '--text',
                        dest='text',
                        required=True,
                        help='Text to classify')
    parser.add_argument('-d', '--dictionary',
                        dest='word2id',
                        required=True,
                        help='Translation table')

    args = parser.parse_args()

    host, port = args.server.split(':')

    return host, port, args.text


def main():
    host, port, text = parse_args()

    # Process string
    # TODO Translate string to numeric

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    with open(text, 'rb') as f:
        data = f.read()

        start = time.time()

        request = predict_pb2.PredictRequest()

        request.model_spec.name = 'cls'
        request.model_spec.signature_name = 'predict_text'
        request.inputs['text'].CopyFrom(make_tensor_proto(data, shape=[1]))

        result = stub.Predict(request, 60.0)

        end = time.time()
        time_diff = end - start

        print(result)
        print('Time elapsed: {}'.format(time_diff))


if __name__ == '__main__':
    main()
