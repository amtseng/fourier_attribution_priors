import torch
import model
import numpy as np

m = model.BinaryTFBindingPredictor(
    input_length=1000,
    input_depth=4,
    num_conv_layers=3,
    conv_filter_sizes=[15, 15, 13],
    conv_stride=1,
    conv_depths=[50, 50, 50],
    max_pool_size=40,
    max_pool_stride=40,
    num_fc_layers=2,
    fc_sizes=[50, 15],
    num_outputs=5,
    batch_norm=True,
    conv_drop_rate=0.0,
    fc_drop_rate=0.2
)

x = torch.tensor(np.random.random([10, 4, 1000])).float()
y = torch.tensor(np.random.randint(2, size=[10, 5])).float()
out = m(x)
loss = m.loss(y, out)
