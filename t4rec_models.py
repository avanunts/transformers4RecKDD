from transformers4rec import torch as tr

from transformers4rec.torch.features.sequence import TabularSequenceFeatures
from transformers4rec.torch.ranking_metric import MeanReciprocalRankAt, RecallAt

metrics = [
    MeanReciprocalRankAt(top_ks=[20, 40], labels_onehot=True),
    RecallAt(top_ks=[20, 40], labels_onehot=True)
]

'''
args: 
- seq_length (int)
- embedding_dims (dict: feature name -> dim, e.g. {'item-list': 64})
- masking (str, e.g. 'clm', 'mlm')
- xlnet_d_model (int)
- xlnet_n_head (int)
- xlnet_n_layer (int)
- weight_tying (boolean)
'''


def xl_net_model(args):
    inputs = TabularSequenceFeatures.from_schema(
        args['schema'],
        aggregation='concat',
        max_sequence_length=args['seq_length'],
        embedding_dims=args['embedding_dims'],
        masking=args['masking'],
    )

    # Define XLNetConfig class and set default parameters for HF XLNet config
    transformer_config = tr.XLNetConfig.build(
        d_model=args['xlnet_d_model'],
        n_head=args['xlnet_n_head'],
        n_layer=args['xlnet_n_layer'],
        total_seq_length=args['seq_length']
    )

    # Define the model block including: inputs, masking, projection and transformer block.
    body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock([args['xlnet_d_model']]),
        tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )

    # Define a head related to next item prediction task
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=args['weight_tying'], metrics=metrics),
        inputs=inputs,
    )

    # Get the end-to-end Model class
    model = tr.Model(head)
    return model
