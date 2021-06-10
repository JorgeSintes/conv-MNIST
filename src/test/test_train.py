import src.models.model as the_model
from torch import nn, optim
import torch

class Test_train:

    def MNIST(input_path, batch_size):
        train_imgs, train_labels = torch.load(input_path)
        train_imgs.unsqueeze_(1)
        d_dataset = torch.utils.data.TensorDataset(train_imgs / 255.0, train_labels)
        dataset = torch.utils.data.DataLoader(d_dataset, batch_size=batch_size, shuffle=True)

        return dataset

    def test_negative_loss(self):

        test_model = the_model.MyAwesomeModel()


        train_input_path = '../../data/processed/training.pt'
        batch_size = 64
        dataset = Test_train.MNIST(train_input_path, batch_size)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(test_model.parameters(), lr=0.001)

        test_model.train()

        for images, labels in dataset:
            optimizer.zero_grad()
            output = test_model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            assert loss.item() > 0


    def test_changing_weights(self):
        test_model = the_model.MyAwesomeModel()

        train_input_path = '../../data/processed/training.pt'
        batch_size = 64
        dataset = Test_train.MNIST(train_input_path, batch_size)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(test_model.parameters(), lr=0.01)


        test_model.train()
        for itr, (images, labels) in enumerate(dataset):

            optimizer.zero_grad()
            output = test_model.forward(images)
            loss = criterion(output, labels)
            before_pass = test_model.linear_1.weight.grad
            loss.backward()
            after_pass = test_model.linear_1.weight.grad
            optimizer.step()

            if itr > 1:
                assert (before_pass.shape[0]*before_pass.shape[1]) == torch.sum(before_pass==after_pass)








