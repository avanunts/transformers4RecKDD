from dataclasses import dataclass


@dataclass(kw_only=True)
class XLNetConfig:
    seq_length: int
    embedding_dims: dict
    masking: str
    xlnet_d_model: int
    xlnet_n_head: int
    xlnet_n_layer: int
    weight_tying: bool = True
    loss: str = 'XE'
    embeddings_layer_norm: bool = False
