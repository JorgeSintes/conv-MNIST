import src.models.model as the_model
import torch
import pytest

class Test_model:

    def MNIST(input_path, batch_size):

        train_imgs, train_labels = torch.load(input_path)
        train_imgs.unsqueeze_(1)
        d_dataset = torch.utils.data.TensorDataset(train_imgs / 255.0, train_labels)
        dataset = torch.utils.data.DataLoader(d_dataset, batch_size=batch_size, shuffle=True)

        return dataset

    def test_model_01(self):
        model = the_model.MyAwesomeModel()
        model.load_state_dict(torch.load('../../models/checkpoint.pth'))

        model.eval()

        test_input_path = '../../data/processed/test.pt'
        batch_size = 64
        dataset = Test_model.MNIST(test_input_path, batch_size)

        for images, labels in dataset:

            output = model.forward(images)

            for im in range(len(images)):
                assert images[im].shape == torch.Size([1, 28, 28])
                assert len(output[im]) == 10


