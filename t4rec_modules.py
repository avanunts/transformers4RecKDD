from torch.nn import LayerNorm


class T4RecLayerNorm(LayerNorm):
    def __init__(self, input_shape):
        self.output_size = input_shape
        super().__init__(normalized_shape=input_shape[-1])

