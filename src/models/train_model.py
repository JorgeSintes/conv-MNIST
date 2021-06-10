# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:30:04 2021

@author: KWesselkamp
"""

import sys
#sys.path.insert(0,'../../')
import argparse
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
from model import MyAwesomeModel
import numpy as np


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def validation(model, testloader, criterion):
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss, accuracy
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        
        # TODO: Implement training loop here
        
        model = MyAwesomeModel()
        
        train_imgs, train_labels = torch.load('../../data/processed/training.pt')
        train_imgs.unsqueeze_(1)
        train_dataset = torch.utils.data.TensorDataset(train_imgs / 255.0, train_labels)
        train_set=torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
          

        test_imgs, test_labels = torch.load('../../data/processed/test.pt')
        test_imgs.unsqueeze_(1)
        test_dataset = torch.utils.data.TensorDataset(test_imgs / 255.0, test_labels)
        test_set = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr= float(args.lr))
        epochs=2
        print_every = 40
        tLoss= []

        steps = 0
        running_loss = 0
        for e in range(epochs):
            
            # Model in training mode, dropout is on
            model.train()
            class_range=np.array([])
            for images, labels in train_set:
            # for itr, (images,labels) in enumerate(train_set):
                np.append(class_range, torch.sort(torch.unique(test_labels)).values.numpy())
                print(class_range)
                steps += 1
                
                # Flatten images into a 784 long vector


                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy = TrainOREvaluate.validation(model, test_set, criterion)

                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss / len(test_set)),
                          "Test Accuracy: {:.3f}".format(accuracy / len(test_set)))

                    tLoss.append(running_loss / print_every)
                    running_loss = 0

                    # Make sure dropout and grads are on for training
                    model.train()
            print(class_range)
        
        print(np.sort(np.unique(class_range)), np.arange(10))
        torch.save(model.state_dict(), '../../models/checkpoint.pth')

        plt.plot(tLoss, label='Training Loss')

        plt.legend()
        plt.savefig('../../reports/figures/training_loss_curve.png')




        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluating arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = MyAwesomeModel()
            model.load_state_dict(torch.load(args.load_model_from))
            model.eval()

            accuracy = 0


            _, test_set = mnist()

           # with torch.no_grad():
            for itr, (images, labels) in enumerate(test_set):
                output = model.forward(images)

                ps = torch.exp(output)

                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Test Accuracy: {:.3f}".format(accuracy / len(test_set)))

        else:
            print("There is no model to evaluate please train first...")




if __name__ == '__main__':
    TrainOREvaluate()