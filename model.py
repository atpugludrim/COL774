# -*- coding: utf-8 -*-
# BASIC CNN MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from skimage import io, transform

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import datetime

class blog:
    def __init__(this):
        if not os.path.isdir("logs"):
            os.mkdir("logs")
    def write(this,x):
        this.term.write(x)
        this.lfile.write(x)
    def flush(this, *args, **kwargs):
        this.term.flush(*args,**kwargs)
        this.lfile.flush(*args,**kwargs)

class olog(blog):
    def __init__(this):
        super().__init__()
        this.term = sys.stdout
        this.lfile = open("logs/log.olog","a")

class elog(blog):
    def __init__(this):
        super().__init__()
        this.term = sys.stderr
        this.lfile = open("logs/log.elog","a")

sys.stdout = olog()
sys.stderr = elog()
now = datetime.datetime.now().strftime("%H:%M:%S, %d %B %Y")
runcom = " ".join(sys.argv)

print("")
print("",file=sys.stderr)
print(runcom)
print(runcom,file=sys.stderr)
print(now,flush=True)
print(now,file=sys.stderr,flush=True)
class Rescale:
  def __init__(this, output_size):
    assert isinstance(output_size, (int, tuple))
    this.output_size = output_size
  def __call__(this, image):
    h, w = image.shape[:2]
    if isinstance(this.output_size, int):
      if h > w:
        new_h, new_w = this.output_size * h / w, this.output_size
      else:
        new_h, new_w = this.output_size, this.output_size * w / h
    else:
      new_h, new_w = this.output_size
    
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(image, (new_h, new_w))
    return img

class ToTensor:
  def __call__(this, image):
    # numpy_image: H x W x C (C is color)
    # torch imagae: C x H x W
    image = image.transpose((2,0,1))
    return torch.FloatTensor(image)

IMAGE_RESIZE = (256,256)
img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

class CaptionsPreprocessing:
  """Preprocess the captions, generate vocab and convert words to tensor tokens
  """
  def __init__(this, captions_file_path):
    this.maximum_cap_len = -float('inf')
    this.captions_file_path = captions_file_path
    this.raw_captions_dict = this.read_raw_captions()
    this.captions_dict = this.process_captions()
    this.vocab = this.generate_vocabulary()
  def read_raw_captions(this):
    """Returns dictionary with raw captions list keyed by image ids (integers)"""
    captions_dict = {}
    with open(this.captions_file_path,"r",encoding='utf-8') as f:
      for img_caption_line in f:
        img_captions = img_caption_line.strip().split('\t')
        captions_dict[img_captions[0]] = img_captions[1]
        l = len(img_captions[1].strip().split())
        if this.maximum_cap_len < l:
            this.maximum_cap_len = l
    return captions_dict
  
  def process_captions(this):
    """ Generate dictionary and other preprocessing"""
    raw_captions_dict = this.raw_captions_dict
    
    # DO THE PROCESSING HERE
    captions_dict = raw_captions_dict
    return captions_dict
  
  def generate_vocabulary(this):
    """ Generate dictionary and other preprocessing"""
    captions_dict = this.captions_dict
    # Generate vocabulary
    return None
  
  def captions_transform(this, img_captions_list):
    """ Generate tensor tokens for the text captions"""
    vocab = this.vocab
    
    # Gnerate tensors

    return torch.zeros(len(img_captions_list),10)

def proc_capt_prox(this):
  raw_captions_dict = this.raw_captions_dict
  captions_dict = raw_captions_dict
  for k in captions_dict:
    ck = captions_dict[k]
    ck = ck + ' xxend'
    while(len(ck.split()) <= this.maximum_cap_len):
        ck = ck+' xxpad'
    captions_dict[k] = 'xxstart ' + ck
  this.maximum_cap_len = this.maximum_cap_len + 2
  return captions_dict

def gen_vocab_proxy(this):
  captions_dict = this.captions_dict
  words = dict()
  for k in captions_dict:
    for token in captions_dict[k].strip().split():
      words[token] = 1
  vocab = {k: v+3 for v, k in enumerate(words)}
  vocab['xxpad'] = 0
  vocab['xxstart'] = 1
  vocab['xxend'] = 2
  vocab['xxunk'] = len(vocab.keys())
  return vocab # vocabulary maps word to integer

def capt_tra_proxy(this, image_caption):
  vocab = this.vocab
  max_ = len(vocab.keys())
  curr_t = []
  for tok in image_caption.strip().split():
      if tok in vocab:
          curr_t.append(vocab[tok])
      else:
          curr_t.append(vocab['xxunk'])
  oh = np.zeros((len(curr_t),max_))
  try:
      oh[np.arange(len(curr_t)),curr_t] = 1
  except:
      print(curr_t)
      print(image_caption)
      raise
  capt_list = torch.FloatTensor(oh)
  return capt_list

CaptionsPreprocessing.process_captions = proc_capt_prox
CaptionsPreprocessing.generate_vocabulary = gen_vocab_proxy
CaptionsPreprocessing.captions_transform = capt_tra_proxy

CAPTIONS_FILE_PATH = 'Train_text.tsv'
captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)

class ImageCaptionsDataset(Dataset):
  def __init__(this, img_dir, captions_dict, img_transform=None, captions_transform=None):
    this.img_dir = img_dir
    this.captions_dict = captions_dict
    this.img_transform = img_transform
    this.captions_transform = captions_transform

    this.image_ids = list(captions_dict.keys())
  def __len__(this):
    return len(this.image_ids)
  
  def __getitem__(this, idx):
    img_name = this.image_ids[idx]
    image = io.imread(img_name)
    captions = this.captions_dict[img_name]
    if this.img_transform:
      image = this.img_transform(image)
    if this.captions_transform:
      captions = this.captions_transform(captions)
    
    sample = {'image':image,'captions':captions}

    return sample

class ImageCaptionsNet(nn.Module):
  def __init__(this,indim,hiddenconvdim,lstmindim,lstmoutdim,batch_size):
    super(ImageCaptionsNet, this).__init__()
    this.indim = indim
    this.hiddenconvdim = hiddenconvdim
    this.lstmindim = lstmindim
    this.lstmoutdim = lstmoutdim
    this.batch_size = batch_size
    # Define architecture here
    # Encoder
    this.convs = nn.ModuleList()
    this.convs.append(nn.Conv2d(indim, hiddenconvdim,(7,7)))
    # THINK OF DECODING USING ALPHABETS INSTEAD OF WORDS
    # THINK ABOUT GIVING FOUR ROTATE INPUTS USING TORCH AND THEN USING ATTENTION
    this.nconvs = 3
    # THINK OF POOLING LAYERS AND SKIP CONNECTIONS
    for _ in range(this.nconvs-1):
        this.convs.append(nn.Conv2d(hiddenconvdim,hiddenconvdim,(7,7)))
    # Attention
    this.flatten = nn.Conv2d(hiddenconvdim,1,(3,3),padding=1,padding_mode='replicate')
    this.att_in = nn.Linear(1,22*22)
    # Decoder
    # Get the output size after running the input through the cnn
    print(lstmoutdim)
    this.rnn = nn.LSTMCell(lstmoutdim,lstmoutdim)
    this.init_h = nn.Linear(22*22,lstmoutdim)
    this.init_c = nn.Linear(22*22,lstmoutdim)
  def forward(this, x):
    image_batch, captions_batch = x
    d = image_batch.device
    # Forward propagations
    for i in range(this.nconvs):
        image_batch = this.convs[i](image_batch)
        image_batch = F.dropout(image_batch,p=0.1,training=this.training)
        image_batch = F.relu_(image_batch)
    image_batch = F.max_pool2d(image_batch,4,11)
    print(image_batch.shape)

    bs,C,w,h = image_batch.size()
    x = this.flatten(image_batch)
    image_batch = image_batch.view(bs,-1,w*h)# flatten
    bs,C,w,h = x.size()
    x = x.view(bs,w*h,-1)
    x = F.softmax(this.att_in(x),dim=2)
    combined = torch.bmm(image_batch,x.permute(0,2,1))

    mean_enc_out = combined.mean(dim=1)
    h = this.init_h(mean_enc_out)
    h = h.view(this.batch_size,this.lstmoutdim)
    c = this.init_c(mean_enc_out)
    c = c.view(this.batch_size,this.lstmoutdim)
    # use lstm cells and decode for every time step (for desired seq len)
    # captions_pred, _ = this.rnn(image_batch*x,(h,c))
    max_len = this.lstmindim
    pred = torch.zeros(this.batch_size,max_len,this.lstmoutdim,device=d)
    # init_input = F.one_hot(torch.tensor([1]),num_classes = this.lstmoutdim).squeeze() # 1 is the start-vector
    init_input = torch.zeros(this.batch_size,this.lstmoutdim,device=d)
    init_input[:,1]=1
    pred[:,0,:] = init_input
    for t in range(1,max_len):
        (h,c) = this.rnn(init_input,(h,c))
        pred[:,t,:] = h# softmax to be done later
        if this.training:
            init_input = captions_batch[:,t,:]
        else:
            init_input = h
    return pred

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
BATCH_SIZE = 16
net = ImageCaptionsNet(3,32,captions_preprocessing_obj.maximum_cap_len,len(captions_preprocessing_obj.vocab.keys()),BATCH_SIZE)
net = net.to(device)

IMAGE_DIR = ''

train_dataset = ImageCaptionsDataset(IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform = img_transform, captions_transform=captions_preprocessing_obj.captions_transform)
NUMBER_OF_EPOCHS = 3
LEARNING_RATE = 1e-1
NUM_WORKERS = 0 # Parallel threads for dataloading
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if torch.cuda.is_available():
    f = True
else:
    f = False
for epoch in range(NUMBER_OF_EPOCHS):
  for batch_idx, sample in enumerate(train_loader):
    optimizer.zero_grad()
    image_batch, captions_batch = sample['image'], sample['captions']
    image_batch, captions_batch = image_batch.to(device), captions_batch.to(device)
    output_captions = net((image_batch, captions_batch))
    #
    captions_ = torch.nonzero(captions_batch)
    #############################################################
    # 1) nll loss takes class as target not one hot vector      #
    # 2) think about just concating all the preds because loss  #
    #    dont care about time steps                             #
    #############################################################
    capt = torch.zeros(BATCH_SIZE,captions_preprocessing_obj.maximum_cap_len,device=device).long()
    capt[captions_[:,0],captions_[:,1]]=captions_[:,2]
    print(captions_batch.shape,output_captions.shape)
    loss = loss_function(output_captions.transpose(1,2), capt)
    loss.backward()
    optimizer.step()
  print("Iteration: ",epoch+1)

with torch.no_grad():
    net.eval()
    vocab = captions_preprocessing_obj.vocab
    inv_vocab = {v:k for k,v in zip(vocab.keys(),vocab.values())}
    max_len = captions_preprocessing_obj.maximum_cap_len
    capts = ['' for _ in range(BATCH_SIZE)]
    for sample in train_loader:
        image_batch = sample['image']
        image_batch = image_batch.to(device)
        pred_capt = net((image_batch,torch.zeros(1,1,device=device)))
        for t in range(max_len):
            p = torch.softmax(pred_capt[:,t,:],dim=1)
            for i in range(BATCH_SIZE):
                pv = int(torch.argmax(p[i,:].view(-1)).item())
                capts[i] += inv_vocab[pv]
        break
    print(capts)
