

# this script can be run in a distributed or single gpu modes:

# 1. multiple gpus distributed run:
# python -m torch.distributed.launch --nproc_per_node=2 ./mimic_lm_distr.py --seed=`shuf -i 0-999 -n1`
# adjust 2 to the number of gpus you want to use
#
# 2. single gpu
# MASTER_ADDR="127.0.0.1" MASTER_PORT="8888" WORLD_SIZE=1 RANK=0 python ./mimic_lm_distr.py --local_rank=0 --seed=`shuf -i 0-999 -n1`


# Now try to run this with more than one GPU.  I will try 2x 1080 first, so want to create a single
# cell that I can later paste into a .py file, since can't run distributed from a notebook.
from fastai.text import *
from fastai.script import *
from fastai.distributed import *

#defaults.cpus = 3          # limit to 3 out of 6 cores to save main memory
torch.cuda.set_device(0)   # Set it to the 2080 TI
#torch.cuda.set_device(1)   # Set it to the 1080; the one used by the monitor
#torch.cuda.set_device(2)   # Set it to the other 1080; not used by the monitor
import gc                   # for garbage collection, though not sure it matters
import datetime             # for time sstamps in the output to log progress
import glob                 # for reading directories of files below
import csv                  # for loading csv files into dictionaries
import spacy                # for over-riding the default tokenizer
from spacy.symbols import ORTH
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'       # enable to cuda debug
import apex.fp16_utils as fp16

# symlink to where your mimic-iii dataset is
path = Path('./mimic')

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

@call_parse
def main(
        local_rank: Param("passed by torch.distributed.launch", int)=None,
        seed:       Param("Random seed", int)=42,
):
    "Distributed training of LM."
    gpu = local_rank
    np.random.seed(seed)

    # distributed setup
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


    #path = datapath4file('/media/DataHD2/Notes_PHI_20190121/notes_dana_hp')
    # Below, I use bptt=72 rather than 70
    data_lm = (TextList.from_csv(path, 'NOTEEVENTS1000.csv', cols='TEXT',
                processor = [TokenizeProcessor(), NumericalizeProcessor(max_vocab=60000)])
                    .split_by_rand_pct()
                    .label_for_lm()
                    .databunch(bs=500, num_workers=4, bptt=72))  # bs=500 works with NOTEEVENTS1000 on the RTX 2080TI
    # Pad the vocab to a multiple of 8 - this along gives another 10% speedup.
    gap = 8-len(data_lm.vocab.itos)%8
    data_lm.vocab.itos += ['xx'+str(i) for i in range(gap)]

    data_lm.save('data_lm_mimic1000_export.pkl')

    # below, I use to_fp16() and set pretrained=False
    # because the pretrained model uses n_hid of 1150, but needs 1144
    # This means that I am forced to pre-train from scratch, but that is probably just as well
    learn = to_fp16(language_model_learner(data_lm, AWD_LSTM, pretrained=False))

    learn = learn.to_distributed(gpu)

    moms = (0.8,0.7)
    t0=datetime.datetime.now()
    print(t0, 'Starting training 10 epochs',flush=True)
    learn.fit_one_cycle(10, slice(1e-2), moms=moms)
    #learn.fit_one_cycle(1, 1e-2)
    t1=datetime.datetime.now()
    print(t1, 'Finished training 10 epoch',flush=True)
    print('duration',t1-t0)
    learn.save('fit_mimic1000')
    learn.save_encoder('fit_mimic1000_enc')
    print(datetime.datetime.now(), 'Finished saving models')
