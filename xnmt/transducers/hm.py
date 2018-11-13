from collections.abc import Sequence
from typing import List

import numpy as np
import dynet as dy

from xnmt import expression_seqs
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.param_collections import ParamManager
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.transducers import base as transducers
from xnmt.persistence import serializable_init, Serializable, Ref, bare


#?? yaml_tags, @register_xnmt_handler, @serializable_init
#?? expr_seq & input format
#?? division between cell & hm for layers / c / h etc

class HM_LSTMCell(object):
    """
    https://arxiv.org/pdf/1609.01704.pdf
    """
    def __init__(self,
                 below_dim,
                 hidden_dim,
                 above_dim,
                 last_layer,
                 a,
                 param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
                 bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
        self.below_dim  = below_dim
        self.hidden_dim = hidden_dim
        self.above_dim  = above_dim
        self.last_layer = last_layer
        self.a = a  #for slope annealing
        model = ParamManager.my_params(self)

        self.p_W_1l_r = model.add_parameters(dim=(hidden_dim*4 + 1, hidden_dim), init=param_init.initializer((hidden_dim*4 + 1, hidden_dim)))
        if not self.last layer:
            self.p_W_2l_td = model.add_parameters(dim=(hidden_dim*4 + 1, above_dim), init=param_init.initializer((hidden_dim*4 + 1, above_dim)))
        self.p_W_0l_bu = model.add_parameters(dim=(hidden_dim*4 + 1, below_dim), init=param_init.initializer((hidden_dim*4 + 1, below_dim)))
        self.p_bias = model.add_parameters(dim=(hidden_dim*4 + 1,), init=bias_init.initializer((hidden_dim*4 + 1,))) #?? cherry: z_bias init to 1

    #?? how to anneal a?
    def hard_sigmoid_slope_anneal(a=1, x):
        tmp = (a * x + 1.0) / 2.0
        output = np.clip(tmp, a_min=0, a_max=1)
        return output

#?? map to this format
#    def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':  
    def forward(self, c, h_below, h, h_above, z, z_below):
        W_1l_r  = dy.parameter(self.p_W_1l_r)
        bias = dy.parameter(self.p_bias)
        
        s_recur = W_11_r * h  #matrix multiply is *, element-wise is dy.cmult
        if not self.last_layer:
            W_2l_td = dy.parameter(self.p_W_2l_td)
            W_0l_bu = dy.parameter(self.p_W_0l_bu)
            s_bottomup = W_01_bu * h_below
            s_topdown  = W_21_td * h_above
        else:
            s_topdown  = np.zeros(s_recur.size())  # in pytorch would do Variable & requires_grad = False, but i don't see the equiv. ?? when can dynet and np things be multiplied ??
            s_bottomup = W_11_r * h
        s_bottomup = dy.cmult(np.tile(z_below, s_bottomup.shape), s_bottomup) #?? need to make parameter or? do i even need the tile? z_below * s_bottomup?
        s_topdown  = dy.cmult(np.tile(z, s_topdown.shape, s_topdown) #?? need to make parameter or?
        
        fslice = s_recur + s_topdown + s_bottomup + bias #?? checkme. bias has same shape as s_recur et al?

        i_ft = dy.pick_range(fslice, 0, self.hidden_dim)
        i_it = dy.pick_range(fslice, self.hidden_dim, self.hidden_dim*2)
        i_ot = dy.pick_range(fslice, self.hidden_dim*2, self.hidden_dim*3)
        i_gt = dy.pick_range(fslice, self.hidden_dim*3, self.hidden_dim*4)
        f_t = dy.logistic(i_ft)  #?? why was there a +1.0 in customlstmseqtransducer
        i_t = dy.logistic(i_it)
        o_t = dy.logistic(i_ot)
        g_t = dy.tanh(i_ut)
        
        z_tilde = hard_sigmoid_anneal(self.a, fslice[self.hidden_dim*4:self.hidden_dim*4+1, :])  #should be hard sigmoid + slope annealing
        z_new = dy.round(z_tilde, gradient_mode="straight_through_gradient")  #use straight-through estimator for gradient: step fn forward, hard sigmoid backward

        #z = z_l,t-1
        #z_below = z_l-1,t
        if z == 1:  #FLUSH
            c_new = dy.cmult(i_t, g_t)
            h_new = dy.cmult(o_t, dy.tanh(c_new))
        elif z_below == 0:  #COPY
            c_new = c
            h_new = h
        else:  #UPDATE
            c_new = dy.cmult(f_t, c) + dy.cmult(i_t * g_t)
            h_new = dy.cmult(o_t, dy.tanh(c_new))
            
        return h_new, c_new, z_new, g_t


class HM_LSTM(object):
    """
    hard-coded to two layers at the moment
    """
    def __init__(self, input_dim, hidden_dim, a):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.a = a  #slope annealing
        
        self.cell_1 = HM_LSTMCell(self.input_dim, self.hidden_dim, self.hidden_dim, self.a, last_layer=False)
        self.cell_2 = HM_LSTMCell(self.hidden_dim, self.hidden_dim, None, self.a, last_layer=True)
        
#?? map to this format
#    def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':  
    def forward(self, inputs, hidden):
        time_steps = 2
        batch_size = 1
        if hidden == None:
            h_t1 = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)] #?? is this [hidden_dim, batch_size] ?
            c_t1 = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
            z_t1 = [dy.zeroes(dim=(1,), batch_size=batch_size)] #?? correct data structure/dims?
            h_t2 = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
            c_t2 = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
            z_t1 = [dy.zeroes(dim=(1,), batch_size=batch_size)]
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden

        z_one = dy.ones(1, batch_size) #?? checkme.
        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        
        for t in range(time_steps):
            #?? h_below inputs = expr_seq?? for layer 1?
            h_t1, c_t1, z_t1, g_t1 = self.cell_1(c=c_t1, h_below=inputs_at_x_t, h=h_t1, h_above=h_t2, z=z_t1, z_below=z_one)
            h_t2, c_t2, z_t2, g_t2 = self.cell_2(c=c_t2, h_below=h_t1, h=h_t2, h_above=None, z=z_t2, z_below=z_t1)
            h_1.append(h_t1)
            h_2.append(h_t2)
            z_1.append(z_t1) #?? checkme. [z_t1]?
            z_2.append(z_t2)
                    
        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        g_t = (g_t1, g_t2)
            
        return h_t1, c_t1, z_t1, h_t2, c_t2, z_t2, g_t1, g_t2
