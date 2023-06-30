from transformers4rec import torch as tr

from torch.nn import CrossEntropyLoss

from transformers4rec.torch.features.sequence import TabularSequenceFeatures
from transformers4rec.torch.ranking_metric import MeanReciprocalRankAt, RecallAt

from . import losses
from . import modules

metrics = [
    MeanReciprocalRankAt(top_ks=[20, 40], labels_onehot=True),
    RecallAt(top_ks=[20, 40], labels_onehot=True)
]

'''
args: (dict)
- seq_length (int)
- embedding_dims (dict: feature name -> dim, e.g. {'item-list': 64})
- masking (str, e.g. 'clm', 'mlm')
- xlnet_d_model (int)
- xlnet_n_head (int)
- xlnet_n_layer (int)
- weight_tying (boolean)
- loss (str): one of 'XE', 'bpr-max', default 'XE'
- layer_norm (bool): if yes, use layer norm before projection, if key not found set to false
schema: (merlin.schema.Schema)
'''


def xlnet_model(config, schema):
    inputs = TabularSequenceFeatures.from_schema(
        schema,
        aggregation='concat',
        max_sequence_length=config.seq_length,
        embedding_dims=config.embedding_dims,
        masking=config.masking,
    )

    # Define XLNetConfig class and set default parameters for HF XLNet config
    transformer_config = tr.XLNetConfig.build(
        d_model=config.xlnet_d_model,
        n_head=config.xlnet_n_head,
        n_layer=config.xlnet_n_layer,
        total_seq_length=config.seq_length
    )

    # Define the model block including: inputs, masking, projection and transformer block.
    if not config.embeddings_layer_norm:
        body = tr.SequentialBlock(
            inputs,
            tr.MLPBlock([config.xlnet_d_model]),
            tr.TransformerBlock(transformer_config, masking=inputs.masking)
        )
    else:
        body = tr.SequentialBlock(
            inputs,
            modules.T4RecLayerNorm(inputs.output_size()),
            tr.MLPBlock([config.xlnet_d_model]),
            tr.TransformerBlock(transformer_config, masking=inputs.masking)
        )

    # Define a head related to next item prediction task
    if config.loss == 'XE':
        loss = CrossEntropyLoss()
    elif config.loss == 'bpr-max':
        loss = losses.BPRMaxLoss()
    else:
        raise ValueError('Loss {} is not implemented'.format(config.loss))
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(loss=loss, weight_tying=config.weight_tying, metrics=metrics),
        inputs=inputs,
    )

    # Get the end-to-end Model class
    model = tr.Model(head)
    return model
