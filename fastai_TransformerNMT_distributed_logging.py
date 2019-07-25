from __future__ import print_function

import argparse
from fastai.text import *
from fastai.script import *
from fastai.distributed import *
from fastprogress import fastprogress
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from tbc import TensorBoardFastAI
from transformer import *
from seq2seq_metrics import *
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


#Linue EU Parliment data for translation
def create_data(path, base:str='fr', targ:str='en'):
    with open(path/'europarl-v7.fr-en.fr') as f: fr = f.read().split('\n')
    with open(path/'europarl-v7.fr-en.en') as f: en = f.read().split('\n')
    df = pd.DataFrame({'fr': [a for a in fr], 'en': [b for b in en]}, columns = ['en', 'fr'])
    df['en'] = df['en'].apply(lambda x:str(x).lower())
    df['fr'] = df['fr'].apply(lambda x:str(x).lower())
    df.en = df.en.astype(str)
    df.fr = df.fr.astype(str)

    data = Seq2SeqTextList.from_df(df, path = path, cols=base)\
                                .split_by_rand_pct(seed=42)\
                                .label_from_df(cols=targ, label_cls=TextList)\
                                .filter_by_func(lambda x,y: len(x) > 60 or len(y) > 60)\
                                .databunch()
    data.save()

def worker(ddp=True):
    # base,targ = 'en','fr'
    name = f'seq2seq_tfrm_{args.base}_{args.targ}'
    gpu = args.local_rank
    bs = 80  # 208:RTX, 128:V100
    epochs = args.epochs
    lr = 1e-3
    # if ddp: lr *= args.proc_per_node  #does not scale well for Transformer.  Why?

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if ddp: dist.init_process_group(backend='nccl', init_method='env://')

    path = Path('europarl/').absolute()
    fastprogress.SAVE_PATH = f'{name}.txt' #Save the output of the progress bar in {name}.txt

    # only download dataset once per machine, sync workers
    if not (path/'data_save.pkl').is_file() and args.local_rank==0: 
        create_data(path,base=args.base, targ=args.targ)
        print(f"DDP: process {rank}/{world_size}")
    if ddp: dist.barrier()  ## sync up so all workers have the data
    torch.cuda.set_device(gpu)

    data = load_data(path, bs=bs)
    data.add_tfm(shift_tfm)
    n_x_vocab,n_y_vocab = len(data.train_ds.x.vocab.itos), len(data.train_ds.y.vocab.itos)
    model = Transformer(n_x_vocab, n_y_vocab, d_model=512)
    model.apply(init_transformer)

    # Writer to tensorboard
    learn = Learner(data, model, metrics=[accuracy, CorpusBLEU(n_y_vocab)], 
                    loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1))
    learn = learn.to_fp16(dynamic=True, clip=2.0)
    if ddp: learn = learn.to_distributed(gpu)

    if (ddp and rank==0) or not ddp:
        ##only a callback for the first one?  Do they all need a callback?
        writer = SummaryWriter(comment=name)
        mycallback = partial(TensorBoardFastAI, writer, track_weight=True, track_grad=True, metric_names=['val loss', 'accuracy','bleu'])
        learn.callback_fns.append(mycallback)

    t0 = datetime.datetime.now();    print(t0, f'Starting training {epochs} epochs',flush=True)
    learn.fit_one_cycle(epochs, lr, div_factor=5)
    t1=datetime.datetime.now();    print(t1, f'Finished training {epochs} epoch',flush=True)
    print('duration',t1-t0)    

    learn.save(Path(f'{name}').absolute(), with_opt=False)

def local_launcher():
    os.system(f'python -m torch.distributed.launch --nproc_per_node={args.proc_per_node} '
              f'fastai_TransformerNMT_distributed_logging.py --mode=worker --proc_per_node={args.proc_per_node}')

def launcher():
    import ncluster

    task = ncluster.make_task(name='fastai_NMT_multi_en_fr',
                              image_name='Deep Learning AMI (Ubuntu) Version 23.0',
                              disk_size=500, #500 GB disk space
                              instance_type='p3.8xlarge') #'c5.large': CPU, p3.2xlarge: one GPU, 8x=4 GPU, 16x=8GPU  
    task.upload('fastai_TransformerNMT_distributed_logging.py')  # send over the file. 
    task.upload('transformer.py')  #helper files
    task.upload('seq2seq_metrics.py')
    task.upload('tbc.py')
    task.run('source activate pytorch_p36')
    task.run('conda install -y -c fastai fastai') 
    task.run('pip install tb-nightly')
    task.run('pip install future')
    # task.run('wget https://s3.amazonaws.com/fast-ai-nlp/giga-fren.tgz && tar -xvf giga-fren.tgz')  ## for Qs dataset
    task.run('mkdir europarl && cd europarl')
    task.run('wget http://www.statmt.org/europarl/v7/fr-en.tgz && tar -xvf fr-en.tgz && cd ~/')  ## for Qs dataset
    task.run(f'python -m torch.distributed.launch --nproc_per_node={args.proc_per_node} '
             f'./fastai_TransformerNMT_distributed_logging.py --mode=worker --proc_per_node={args.proc_per_node} --save-model', stream_output=True)

    name = f'seq2seq_tfrm_{args.base}_{args.targ}' 
    task.download(f'{name}.txt')
    task.download(f'{name}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fastai Transformer NMT Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--base', type=str, default='en',
                        help='base (feature) language i.e. en')
    parser.add_argument('--targ', type=str, default='fr',
                        help='target language i.e. fr')

    args = parser.parse_args()

    if args.mode == 'remote':
        launcher()
    elif args.mode == 'local':
        local_launcher()
    elif args.mode == 'worker':
        worker()
    elif args.mode == 'localworker':
        worker(ddp=False)

