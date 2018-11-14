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
#?? param_init for c,h,z
#?? slope annealing

class HM_LSTMCell(object):
    """
    single layer HM implementation to enable HM_LSTMTransducer
    https://arxiv.org/pdf/1609.01704.pdf
    """
    def __init__(self,
                 below_dim,
                 hidden_dim,
                 above_dim,
                 a,
                 last_layer,
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
        self.p_bias = model.add_parameters(dim=(hidden_dim*4 + 1,), init=bias_init.initializer((hidden_dim*4 + 1,)))

        # to track prev timestep c, h, & z values for this layer
        self.c = dy.zeroes(dim=(network.hidden_dim,)) #?? does (hidden,) take care of batch_size?
        self.h = dy.zeroes(dim=(network.hidden_dim,))
        self.z = dy.ones(dim=(1,))

    #todo: annealing schedule for a: `annealing the slope of the hard binarizer from 1.0 to 5.0 over 80k minibatches'
    def hard_sigmoid_slope_anneal(a=1.0, x):
        tmp = (a * x + 1.0) / 2.0
        output = np.clip(tmp, a_min=0, a_max=1) #?? can i do this to a dynet object? is there another way?
        return output


    #xs = input / h_below
    def transduce(self, h_below: 'expression_seqs.ExpressionSequence', h_above, z_below) -> 'expression_seqs.ExpressionSequence':
        W_1l_r  = dy.parameter(self.p_W_1l_r)
        bias = dy.parameter(self.p_bias)
        
        s_recur = W_11_r * self.h  #matrix multiply is *, element-wise is dy.cmult
        if not self.last_layer:
            W_2l_td = dy.parameter(self.p_W_2l_td)
            W_0l_bu = dy.parameter(self.p_W_0l_bu)
            s_bottomup = W_01_bu * h_below
            s_topdown  = W_21_td * h_above
        else:
            s_topdown  = dy.zeroes(s_recur.dim()) #?? this gets the shape e.g. ((5, 1), 1). do i actually want batch_size as well?
            s_bottomup = W_11_r * self.h
        s_bottomup = z_below * s_bottomup 
        s_topdown  = self.z * s_topdown  #will be zeros if last_layer. is this right, or should z=1 in this case ??
        
        fslice = s_recur + s_topdown + s_bottomup + bias #?? checkme. bias has same shape as s_recur et al? [4*hidden+1, batch_size]?

        i_ft = dy.pick_range(fslice, 0, self.hidden_dim)
        i_it = dy.pick_range(fslice, self.hidden_dim, self.hidden_dim*2)
        i_ot = dy.pick_range(fslice, self.hidden_dim*2, self.hidden_dim*3)
        i_gt = dy.pick_range(fslice, self.hidden_dim*3, self.hidden_dim*4)
        f_t = dy.logistic(i_ft + 1.0)  #+1.0 bc a paper said it was better to init that way (matthias)
        i_t = dy.logistic(i_it)
        o_t = dy.logistic(i_ot)
        g_t = dy.tanh(i_ut)
        
        z_tilde = hard_sigmoid_anneal(self.a, fslice[self.hidden_dim*4:self.hidden_dim*4+1, :])  #should be hard sigmoid + slope annealing
        z_new = dy.round(z_tilde, gradient_mode="straight_through_gradient")  #use straight-through estimator for gradient: step fn forward, hard sigmoid backward

        #z = z_l,t-1
        #z_below = z_l-1,t
        if self.z == 1: #FLUSH
            c_new = dy.cmult(i_t, g_t)
            h_new = dy.cmult(o_t, dy.tanh(c_new))
        elif z_below == 0: #COPY
            c_new = self.c
            h_new = self.h
        else: #UPDATE
            c_new = dy.cmult(f_t, self.c) + dy.cmult(i_t * g_t)
            h_new = dy.cmult(o_t, dy.tanh(c_new))
        
        self.c = c_new
        self.h = h_new
        self.z = z_new

        return h_new, z_new


class HM_LSTMTransducer(transducers.SeqTransducer, Serializable):
    """
    hard-coded to two layers at the moment
    """
    yaml_tag = '!HM_LSTMTransducer'
    @register_xnmt_handler
    @serializable_init
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 a,
                 bottom_layer=None,
                 top_layer=None):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.a = a  #for slope annealing
        self.bottom_layer = HM_LSTMCell(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        above_dim=hidden_dim,
                                        a=a,
                                        last_layer=False)
        self.top_layer    = HM_LSTMCell(input_dim=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        above_dim=None,
                                        a=a,
                                        last_layer=True)
        

    @handle_xnmt_event
    def on_start_sent(self, *args, **kwargs):
        self._final_states = None

    def get_final_states(self):
        assert self._final_states is not None, "transduce() must be invoked before get_final_states()"
        return self._final_states
    

    def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':  
        
        batch_size = xs[0][0].dim()[1]
        h_bot = []
        h_top = []
        z_bot = []
        z_top = []

        #?? checkme. want to init z to ones? (cherry paper)
        z_one   = dy.ones(1, batch_size)
        h_t_bot = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
        z_t_bot = [dy.zeroes(dim=(1,), batch_size=batch_size)]
        h_t_top = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
        z_t_top = [dy.zeroes(dim=(1,), batch_size=batch_size)]
        
        for i, x_t in enumerate(xs):
            h_t_bot, z_t_bot = self.bottom_layer(h_below=x_t, h=h_t_bot, h_above=h_t_top, z=z_t_bot, z_below=z_one) #uses h_t_top from layer above@previous time step, h_t_bot and z_t_bot from previous time step
            h_t_top, z_t_top = self.top_layer(h_below=h_t_bot, h=h_t_top, h_above=None, z=z_t_top, z_below=z_t_bot) #uses z_t_bot and h_t_bot from previous layer call, h_t_top and z_t_top from previous time step
            
            h_bot.append(h_t_bot)
            z_bot.append(z_t_bot)
            h_top.append(h_t_top)
            z_top.append(z_t_top)

        #?? final state should be h_top[-1], or?
        self._final_states = [transducers.FinalTransducerState(h_top[-1])]
        return h_top
