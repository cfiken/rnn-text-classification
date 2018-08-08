from typing import List, NamedTuple, Dict

class Config(NamedTuple):
    num_units: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_outputs: int = 10000
    embedding_size = 512
    batch_size: int = 128
    max_length: int = 50
    dropout_in_rate: float = 0.1
    dropout_out_rate: float = 0.2
    learning_rate: float = 0.001
    grad_clip: float = 5.0
    is_layer_norm: bool = False
    checkpoint_dir = './checkpoints/'
    data_path: str = './data/'
    log_dir: str = './logs/'
    
    def to_log_dir(self) -> str:
        return self.log_dir + 'layers={}/units={}/lr={}'.format(self.num_layers, self.num_units, self.learning_rate)
    
    def to_ckpt_path(self) -> str:
        layernorm = 'T' if self.is_layer_norm else 'F'
        return self.checkpoint_dir + 'l{}_u{}_lr{}_ln{}_model.ckpt'.format(self.num_layers, self.num_units, self.learning_rate, layernorm)
