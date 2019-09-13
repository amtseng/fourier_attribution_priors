import torch
import sacred
import model.models as models
import model.util as util
import model.train_binary_model as train
import numpy as np

ex = sacred.Experiment("ex", ingredients=[
])

@ex.automain
def create_model():
    bin_model = models.BinaryTFBindingPredictor(
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
        num_outputs=1,
        batch_norm=True,
        conv_drop_rate=0.0,
        fc_drop_rate=0.2
    )

    return bin_model


class TestDataset(torch.utils.data.IterableDataset):
    def __init__(self, x_list, y_list):
        assert len(x_list) == len(y_list)
        self.x_list = x_list
        self.y_list = y_list
        self.size = len(x_list)

    def __iter__(self):
        return ((self.x_list[i], self.y_list[i]) for i in range(self.size))

    def __len__(self):
        return self.size

    def on_epoch_start(self):
        pass


def main():
    model = models.BinaryTFBindingPredictor(
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
        num_outputs=1,
        batch_norm=True,
        conv_drop_rate=0.0,
        fc_drop_rate=0.2
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = np.random.random([10, 1000, 4])
    y = np.random.randint(2, size=[10, 1])
    dataset = TestDataset([x], [y])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, collate_fn=lambda x: x
    )

    if torch.cuda.is_available:
        torch.cuda.empty_cache()  # Clear GPU memory

    train_epoch_loss = train.train_epoch(
        data_loader, model, optimizer
    )

    val_epoch_loss, pred_vals, true_vals = train.eval_epoch(
        data_loader, model, -1
    )

    util.save_model(model, "test.pt")
    x = util.restore_model(models.BinaryTFBindingPredictor, "test.pt")
