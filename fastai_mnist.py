"""Usage
pip install -r requirements.txt

# run locally
python mnist.py

# run remotely
# set your AWS_DEFAULT_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
python mnist.py --remote
"""

from __future__ import print_function
import argparse
from fastai.vision import *

def worker():
    path = untar_data(URLs.MNIST_SAMPLE)
    data = ImageDataBunch.from_folder(path)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit(args.epochs)
    if args.save_model:
        learn.save(Path('mnist_example').absolute())

def launcher():
    import ncluster

    task = ncluster.make_task(name='fastai_mnist',
                              image_name='Deep Learning AMI (Ubuntu) Version 23.0',
                              instance_type='p3.2xlarge') #'c5.large')
    task.upload('fastai_mnist.py')  # send over the file. 
    task.run('source activate pytorch_p36')
    task.run('conda install -y -c fastai fastai')  ##install fastai
    task.run('python fastai_mnist.py --save-model', stream_output=True)
    task.download('mnist_example.pth')  ## download the model weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fastai MNIST Example')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--remote', action='store_true', default=False,
                        help='run training remotely')
    args = parser.parse_args()

    if args.remote:
        launcher()
    else:
        worker()
