from torch.nn import LayerNorm


class T4RecLayerNorm(LayerNorm):
    def __init__(self, input_shape):
        self.output_size_var = input_shape
        super().__init__(normalized_shape=input_shape[-1])

    def output_size(self, input_size=None):
        if input_size is not None:
            return input_size
        return self.output_size_var

