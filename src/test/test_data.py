import torch
import numpy as np
import pytest


class Test_data:
    def MNIST(input_path, batch_size):

        train_imgs, train_labels = torch.load(input_path)
        train_imgs.unsqueeze_(1)
        d_dataset = torch.utils.data.TensorDataset(train_imgs / 255.0, train_labels)
        dataset = torch.utils.data.DataLoader(d_dataset, batch_size=batch_size, shuffle=True)

        return dataset

    @pytest.mark.parametrize("test_input,expected", [("../../data/processed/training.pt", 60000), ("../../data/processed/test.pt", 10000)])
    def test_train_dataset_01(self,test_input,expected):

        batch_size = 64
        dataset = Test_data.MNIST(test_input, batch_size)

        length_dataset = 0
        for itr, (test_images, test_labels) in enumerate(dataset):
            length_dataset +=len(test_images)

        if "train" in test_input:
            assert length_dataset == expected

        if "test" in test_input:
            assert length_dataset == expected

    @pytest.mark.parametrize("test_input,expected",
                             [("../../data/processed/training.pt", torch.Size([1,28,28])), ("../../data/processed/test.pt", torch.Size([1,28,28]))])
    def test_train_dataset_02(self, test_input, expected):


        batch_size = 64
        dataset = Test_data.MNIST(test_input, batch_size)

        for images, labels in dataset:
            for im in range(len(images)):
                assert images[im].shape == expected

    @pytest.mark.parametrize("test_input,expected",
                             [("../../data/processed/training.pt", 10), ("../../data/processed/test.pt", 10)])
    def test_train_dataset_03(self, test_input, expected):

        batch_size = 64
        dataset = Test_data.MNIST(test_input, batch_size)

        dataset_labes= np.zeros(10)

        for images, labels in dataset:
            dataset_labes[labels] =+1

        assert np.count_nonzero(dataset_labes) == expected












