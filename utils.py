import sys, os 
import logging 
import shutil 
import json 
import torch 

"""Some common utils for training """
class ParamParser(object):
    def __init__(self, param_path):
        super(ParamParser, self).__init__()
        with open(param_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def save(self, json_path):
        with open(json_path, 'w') as of:
            json.dump(self.__dict__, of, indent=4)
    
    @property
    def dict(self):
        return self.__dict__

class AverageMeter(object):
    """
    A class to keep track of a quantity
    """
    def __init__(self):
        self._val = 0 
        self._step = 0 
    
    def update(self, val):
        self._val += val 
        self._step += 1 
    
    def __call__(self):
        return self._val/float(self._step)


def setLogger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s:%(message)s'))
        logger.addHandler(file_handler)

        # log to console as well 
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)


def save_checkpoint(state_dict, isBest, ckpt_dir):
    """
    Saves the checkpoint in the ckpt_dir as model.pth.tar. If isBest 
    is true then a model is also saved as best.pth.tar.
    """
    path = os.path.join(ckpt_dir, 'last.pth.tar')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save(state_dict, path)

    if isBest:
        shutil.copyfile(path, os.path.join(ckpt_dir, 'best.pth.tar'))



def load_checkpoint(ckpt, net, optimizer=None):
    """
    Load a checkpoint into a network. 
    net: nn.Module

    The state_dict must contain the following keys:
    {
        "epoch": xx ,
        "state_dict": net.state_dict, 
        "optimimzer": optim.state_dict
    }
    """
    if not os.path.exists(ckpt):
        print("The given path doesn't exist:{}".format(ckpt))
    checkpt = torch.load(ckpt)
    print(checkpt.keys())
    net.load_state_dict(checkpt['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpt['optimizer'])

    
    
def save_dict_to_json(Dict, json_file):
    with open(json_file, 'w') as of:
        wr_dict  = {k:float(v) for k,v in Dict.items()}
        json.dump(wr_dict, of, indent=4)
