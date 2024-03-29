"""
Train Wiki103 LM from scratch using parameters that are sent.

Raw file can be pulled from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip

Usage
pip install -r requirements.txt

# run locally
python mnist.py

# run remotely
# set your AWS_DEFAULT_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
python mnist.py --remote
"""
from __future__ import print_function
import argparse
from fastai.text import *
from fastai.script import *
from fastprogress import fastprogress

#Functions to parse WT103 in separate articles
def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0

def read_file(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', UNK)
    articles.append(current_article)
    return np.array(articles)

def create_data(path):
    train = read_file(path/'wiki.train.tokens')
    valid = read_file(path/'wiki.valid.tokens')
    test =  read_file(path/'wiki.test.tokens')
    all_texts = np.concatenate([valid, train, test])
    df = pd.DataFrame({'texts':all_texts})
    del train ; del valid ; del test #Free RQM before tokenizing
    data = (TextList.from_df(df, path, cols='texts')
                    .split_by_idx(range(0,60))
                    .label_for_lm()
                    .databunch(bptt=80*4))
    data.save()

def worker():
    ## build data
    name = 'test1'
    gpu = 0
    bs, bptt = 128,80  ## seems to be max on V100
    backwards = False
    drop_mult = 1.
    epochs = 10
    lr = 1e-3
    wd = 0.1

    path = Path('wikitext-103/').absolute()
    fastprogress.SAVE_PATH = f'{name}.txt' #Save the output of the progress bar in {name}.txt
    if not (path/'data_save.pkl').is_file(): create_data(path)
    torch.cuda.set_device(gpu)
    data = load_data(path, bs=bs, bptt=bptt, backwards=backwards)
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, pretrained=False,
                                   metrics=[accuracy, Perplexity()])
    learn = learn.to_fp16(clip=0.1)

    learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7), div_factor=10, wd=wd)

    learn = learn.to_fp32()
    learn.save(Path(f'{name}').absolute(), with_opt=False)
    learn.data.vocab.save(path/f'{name}_vocab.pkl')

    ## prior for MNIST
    # path = untar_data(URLs.MNIST_SAMPLE)
    # data = ImageDataBunch.from_folder(path)
    # learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    # learn.fit(args.epochs)
    # if args.save_model:
    #     learn.save(Path('mnist_example').absolute())

def launcher():
    import ncluster

    task = ncluster.make_task(name='fastai_wk103',
                              image_name='Deep Learning AMI (Ubuntu) Version 23.0',
                              instance_type='p3.2xlarge') #'c5.large': CPU, p3.2xlarge:GPU 
    task.upload('fastai_wk103.py')  # send over the file. 
    task.run('source activate pytorch_p36')
    task.run('conda install -y -c fastai fastai')  ##install fastai
    ## get wiki103 and unzip
    task.run('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip && unzip wikitext-103-v1.zip')

    ## take params from `args` and use in this call here
    ## get this call from a text file?  database query?  Something to specify the tests we are interested in running.
    task.run('python fastai_wk103.py --save-model', stream_output=True)

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
