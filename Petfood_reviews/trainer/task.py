
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from . import model

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--train_data_path',
        help='can be a local path or a GCS url (gs://...)',
        required=True
    )
    parser.add_argument(
        '--eval_data_path',
        help='can be a local path or a GCS url (gs://...)',
        required=True
    )
    parser.add_argument(
        '--model_type',
        help='any of the available models: CNN, LSTM, and BiDirect',
        choices=['CNN', 'LSTM', 'BiDirect'],
        required=True
    )
    parser.add_argument(
        '--embedding_path',
        help='OPTIONAL: can be a local path or a GCS url (gs://...). \
              Download from: https://nlp.stanford.edu/projects/glove/',
    )
    parser.add_argument(
        '--num_epochs',
        help='number of times to go through the data, default=10',
        default=10,
        type=float
    )
    parser.add_argument(
        '--batch_size',
        help='number of records to read during each training step, default=128',
        default=128,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='learning rate for gradient descent, default=.001',
        default=.001,
        type=float
    )
    parser.add_argument(
        '--filters',
        help='number of output dimension of the CNN layers, default=64',
        default=64,
        type=int
    )
    parser.add_argument(
        '--dropout_rate',
        help='percentage of input to drop at Dropout layers, default=.2',
        default=.2,
        type=float
    )   
    parser.add_argument(
        '--embedding_dim',
        help='dimension of the embedding vectors, default=200',
        default=200,
        type=int
    )           
    parser.add_argument(
        '--kernel_size',
        help='length of the convolution window, default=3',
        default=3,
        type=int
    )                              
    parser.add_argument(
        '--pool_size',
        help='factor by which to downscale input at MaxPooling layer, default=3',
        default=3,
        type=int
    )                     
  
    
    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')
    
    model.train_and_evaluate(output_dir, hparams)
