from __future__ import print_function

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
import argparse
from fastai.text import *
from fastai.script import *
from fastai.distributed import *
from fastprogress import fastprogress
import torch.distributed as dist

import datetime             # for timestamps in the output to log progress

# temp workaround that adds find_unused_parameters=True to DistributedDataParallel call - otherwise things crash with pretrained=False (but works fine with pretrained=True)
def on_train_begin_workaround(self, **kwargs):
    self.learn.model = DistributedDataParallel(self.model, device_ids=[self.cuda_id], output_device=self.cuda_id, find_unused_parameters=True)
    shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(self.data.train_dl, 'init_kwargs') else True
    self.old_train_dl,self.data.train_dl,self.train_sampler = self._change_dl(self.data.train_dl, shuffle)
    if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
        self.old_valid_dl,self.data.valid_dl,self.valid_sampler = self._change_dl(self.data.valid_dl, shuffle)
    self.rank = rank_distrib()
    self.recorder.silent = (self.rank != 0)

DistributedTrainer.on_train_begin = on_train_begin_workaround


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
    df = pd.DataFrame({'texts':all_texts}) #.head(500)  ## test set, small number
    del train ; del valid ; del test #Free RQM before tokenizing
    data = (TextList.from_df(df, path, cols='texts')
                    .split_by_idx(range(0,60))
                    .label_for_lm()
                    .databunch(bptt=80))
    data.save()

def worker(ddp=True):
    name = 'test1'
    gpu = args.local_rank
    bs, bptt = 128,80  # 256:RTX, 128:V100
    backwards = False
    drop_mult = 1.
    epochs = args.epochs
    lr = 1e-3
    if ddp: lr *= args.proc_per_node
    wd = 0.1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if ddp: dist.init_process_group(backend='nccl', init_method='env://')

    path = Path('wikitext-103/').absolute()

    # only download dataset once per machine, sync workers
    if not (path/'data_save.pkl').is_file() and args.local_rank==0: 
        create_data(path)
        print(f"DDP: process {rank}/{world_size}")

    if ddp: dist.barrier()  ## sync up so all workers have the data

    torch.cuda.set_device(gpu)

    data = load_data(path, bs=bs, bptt=bptt, backwards=backwards)
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, pretrained=False,
                                   metrics=[accuracy, Perplexity()])
    learn = learn.to_fp16(clip=0.1)
    if ddp: learn = learn.to_distributed(gpu)

    t0 = datetime.datetime.now();    print(t0, f'Starting training {epochs} epochs',flush=True)

    learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7), div_factor=10, wd=wd)

    t1=datetime.datetime.now();    print(t1, f'Finished training {epochs} epoch',flush=True)
    print('duration',t1-t0)    

    learn = learn.to_fp32()
    learn.save(Path(f'{name}').absolute(), with_opt=False)
    learn.data.vocab.save(path/f'{name}_vocab.pkl')


def local_launcher():
    os.system(f'python -m torch.distributed.launch --nproc_per_node={args.proc_per_node} '
              f'fastai_wk103_distributed.py --mode=worker --proc_per_node={args.proc_per_node}')

def launcher():
    import ncluster

    task = ncluster.make_task(name='fastai_wk103_multi',
                              image_name='Deep Learning AMI (Ubuntu) Version 23.0',
                              instance_type='p3.8xlarge') #'c5.large': CPU, p3.2xlarge: one GPU,  
    task.upload('fastai_wk103_distributed.py')  # send over the file. 
    task.run('source activate pytorch_p36')
    task.run('conda install -y -c fastai fastai')  ##install fastai
    ## get wiki103 and unzip
    task.run('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip && unzip wikitext-103-v1.zip')
    task.run(f'python -m torch.distributed.launch --nproc_per_node={args.proc_per_node} '
             f'./fastai_wk103_distributed.py --mode=worker --proc_per_node={args.proc_per_node} --save-model', stream_output=True)
    # task. ## get the file(s)?

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fastai MNIST Example')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--proc_per_node',type=int, default=4, 
                        help='number of processes per machine')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local_rank set for distributed training')
    parser.add_argument('--mode', default='localworker', 
                        choices=['remote', 'local', 'worker', 'localworker'],
                        help="local: spawn multiple processes locally, remote: launch multiple machines/processes on AWS, worker: DDP aware single process process version, localworker: standalone single process version")

    args = parser.parse_args()

    # if args.remote:
    #     launcher()
    # else:
    #     worker()
    if args.mode == 'remote':
        launcher()
        # remote_launcher()
    elif args.mode == 'local':
        local_launcher()
    elif args.mode == 'worker':
        worker()
    elif args.mode == 'localworker':
        worker(ddp=False)

