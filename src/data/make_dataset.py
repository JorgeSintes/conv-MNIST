# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import torch
import shutil


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    input_filepath='..\..\data'   
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # Download and load the training data
    trainset = datasets.MNIST(input_filepath, download=True, train=True, transform=transform)

    # Download and load the test data
    testset = datasets.MNIST(input_filepath, download=True, train=False, transform=transform)

    #replace the MNIST folder
    shutil.rmtree('../../data/raw')
    shutil.move('../../data/MNIST/raw', '../../data/')

    shutil.rmtree('../../data/processed')
    shutil.move('../../data/MNIST/processed', '../../data/')
    
    shutil.rmtree('../../data/MNIST')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
