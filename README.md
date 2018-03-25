## Detecting Toxic Comments
### Introduction
CNN and LSTM models for text classification. The model is tested on a multi-label 
classification task with Wikimedia comments dataset. The model achieved an 
AUROC of 0.896 with randomly initialized word embeddings; using FastText, 
the AUC is 0.972 with Kim Yoon's CNN, and 0.983 with a stacked LSTM with attention.

### Usage

#### Training
To train with default layer configurations
```bash
python training/train.py --data dataset.csv --vocab 30000 --embedding 300 --mode cnn
```
where vocab flag is for specifying vocabulary size and embedding embedding 
size. There are three modes: use 'cnn' for training CNN for 
classification, 'lstm' for training LSTM for classification, 'emb' for training word embeddings, and 'test' for testing
a trained model.

To train with a pre trained word vector file, use the 'vector' flag:
```bash
python training/train.py --data dataset.csv --vocab 30000 --embedding 300 --mode lstm --vector fasttext.vec
```
You can also optionally add a tsv metadata file for TensorBoard projector using the `metadata` flag.

#### Use Deployed model
Make requests to the deployed saved model:
```bash
python training/client.py --server 35.227.88.30:9000 -d "metadata/word2id.pickle" -t "Enter your potential abusive text here."
```
Output is a JSON file:
```
outputs {
  key: "output"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 6
      }
    }
    float_val: 1.0
    float_val: 0.0
    float_val: 1.0
    float_val: 0.0
    float_val: 0.0
    float_val: 0.0
  }
}
```
Each of the six `float_val`s represents toxic, severe_toxic, obscene, threat, insult, identity_hate.

### Custom CNN layers
You can also change the layer configuration if you decide to write your 
own code for training and testing, by providing values to `layer_config` 
and `fully_conn_config` attributes to the ToxicityCNN object. `layer_config` 
is a list and follows the structure: 
```
[
    [
        # Parellel layer 1
        [ksize, stride, out_channels, pool_ksize, pool_stride],
    ],
    [
        # Parellel layer 2
        [ksize, stride, out_channels, pool_ksize, pool_stride],
    ],
]
```
For Example, a configuration like this:
```pythonstub
[
    # Convolution layer configuration
    # ksize, stride, out_channels, pool_ksize, pool_stride
    [
        [2, 1, 256, 59, 1],
    ],
    [
        [3, 1, 256, 58, 1],
    ],
    [
        [4, 1, 256, 57, 1],
    ],
    [
        [5, 1, 256, 56, 1],
    ],
]
```
represents a structure like this:
![config](readme_media/config.png)
