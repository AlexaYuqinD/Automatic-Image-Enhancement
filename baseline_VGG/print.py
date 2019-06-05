# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:57:34 2019

@author: THINKPAD
"""

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("checkpoints\a0002_a0002a_style_transfer\checkpoint",  all_tensors=True, tensor_name='content loss',all_tensor_names=False)
