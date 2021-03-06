import torch
import sacred
import model.profile_models as profile_models
import model.util as util
import numpy as np

ex = sacred.Experiment("ex", ingredients=[
])

def create_model():
    prof_model = profile_models.ProfileTFBindingPredictor(
        input_length=1346,
        input_depth=4,
        profile_length=1000,
        num_tasks=3,
        num_dil_conv_layers=7,
        dil_conv_filter_sizes=([21] + ([3] * 6)),
        dil_conv_stride=1,
        dil_conv_dilations=[2 ** i for i in range(7)],
        dil_conv_depths=([64] * 7),
        prof_conv_kernel_size=75,
        prof_conv_stride=1
    )

    return prof_model


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

model = None
prof, count = None, None
@ex.automain
def main():
    global model, prof, count
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    model = create_model()

    model = model.to(device)

    np.random.seed(20191013)
    x = np.random.randint(2, size=[10, 4, 1346])
    y = (
        np.random.randint(5, size=[10, 3, 2, 1000]),
        np.random.randint(5, size=[10, 3, 2, 1000])
    )

    # dataset = TestDataset([x], [y])
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=None, collate_fn=lambda x: x
    # )

    torch.autograd.set_detect_anomaly(True)

    input_seq = util.place_tensor(torch.tensor(x)).float()
    tf_prof = util.place_tensor(torch.tensor(y[0])).float()
    cont_prof = util.place_tensor(torch.tensor(y[1])).float()

    pred_prof, pred_count = model(input_seq, cont_prof)

    loss = model.correctness_loss(tf_prof, pred_prof, pred_count, 1)

    loss.backward()
