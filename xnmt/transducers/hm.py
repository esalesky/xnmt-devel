from collections.abc import Sequence
from typing import List

import numpy as np
import dynet as dy

from xnmt import expression_seqs, param_initializers
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.param_collections import ParamManager
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.transducers import base as transducers
from xnmt.persistence import serializable_init, Serializable, Ref, bare


#?? yaml_tags / save_processed arg / serializable for HMLSTM. did not want it to be callable from yaml; but error messages led me to the current setup. tried to copy CustomLSTMSeqTransducer but got error messages
#?? param_init for c,h,z
#?? slope annealing

#todo: annealing schedule for a: `annealing the slope of the hard binarizer from 1.0 to 5.0 over 80k minibatches'
def hard_sigmoid_anneal(zz, a=1.0):
    tmp = ((a * zz) + 1.0) / 2.0 #todo: clip to [0,1]
    return tmp
##    output = np.clip(tmp, a_min=0, a_max=1) #?? can i do this to a dynet object? [no]. is there another way? [yes, tbd non-messy way]
#    if max(tmp.value()) > 1:
#        return 
#    elif min(tmp.value()) < 0:
#        return 
#    else:
#        return tmp

class HMLSTMCell(transducers.SeqTransducer, Serializable):
    """
    single layer HM implementation to enable HM_LSTMTransducer
    https://arxiv.org/pdf/1609.01704.pdf
    """
    yaml_tag = '!HMLSTMCell'
    @serializable_init
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 above_dim,
                 a,
                 hier,
                 last_layer,
                 param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
                 bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.above_dim  = above_dim
        self.last_layer = last_layer
        self.a = a  #for slope annealing
        self.hier = hier
#        self.save_processed_arg("last_layer", self.last_layer)
        model = ParamManager.my_params(self)

        self.p_W_1l_r = model.add_parameters(dim=(hidden_dim*4 + 1, hidden_dim), init=param_init.initializer((hidden_dim*4 + 1, hidden_dim)))
        if not self.last_layer:
            self.p_W_2l_td = model.add_parameters(dim=(hidden_dim*4 + 1, above_dim), init=param_init.initializer((hidden_dim*4 + 1, above_dim)))
        self.p_W_0l_bu = model.add_parameters(dim=(hidden_dim*4 + 1, input_dim), init=param_init.initializer((hidden_dim*4 + 1, input_dim)))
        self.p_bias = model.add_parameters(dim=(hidden_dim*4 + 1,), init=bias_init.initializer((hidden_dim*4 + 1,)))

        # to track prev timestep c, h, & z values for this layer
        self.c = None
        self.h = None
        self.z = None


    #xs = input / h_below
    def transduce(self, h_below: 'expression_seqs.ExpressionSequence', h_above, z_below) -> 'expression_seqs.ExpressionSequence':
        if self.c == None:
            self.c = dy.zeroes(dim=(self.hidden_dim,)) #?? does (hidden,) take care of batch_size?
        if self.h == None:
            self.h = dy.zeroes(dim=(self.hidden_dim,))
        if self.z == None:
            self.z = dy.ones(dim=(1,))

        W_1l_r  = dy.parameter(self.p_W_1l_r)
        bias = dy.parameter(self.p_bias)
        h = dy.parameter(self.h)

        s_recur = W_1l_r * h #matrix multiply is *, element-wise is dy.cmult. CURRERROR: stale expression
        if not self.last_layer:
            W_2l_td = dy.parameter(self.p_W_2l_td)
            W_0l_bu = dy.parameter(self.p_W_0l_bu)
            s_bottomup = W_0l_bu * h_below #?? this is becoming (2049,). does it need to be (2049,1) to do scalar * matrix?
            s_topdown  = W_2l_td * h_above
        else:
            s_topdown  = dy.zeroes(s_recur.dim()[0][0],) #?? this gets the shape e.g. ((5, 1), 1). do i actually want batch_size as well?
            s_bottomup = W_1l_r * h
        s_bottomup = dy.cmult(z_below,s_bottomup) #to handle batched scalar * matrix -> e.g. (1x10, 2049x10)
        s_topdown  = dy.cmult(self.z,s_topdown)  #will be zeros if last_layer. is this right, or should z=1 in this case ??
        
        fslice = s_recur + s_topdown + s_bottomup + bias #?? checkme. bias has same shape as s_recur et al? [4*hidden+1, batch_size]?

        i_ft = dy.pick_range(fslice, 0, self.hidden_dim)
        i_it = dy.pick_range(fslice, self.hidden_dim, self.hidden_dim*2)
        i_ot = dy.pick_range(fslice, self.hidden_dim*2, self.hidden_dim*3)
        i_gt = dy.pick_range(fslice, self.hidden_dim*3, self.hidden_dim*4)
        f_t = dy.logistic(i_ft + 1.0)  #+1.0 bc a paper said it was better to init that way (matthias)
        i_t = dy.logistic(i_it)
        o_t = dy.logistic(i_ot)
        g_t = dy.tanh(i_gt)


        #z * normal_update + (1-z)*copy: ie, when z_below is 0, z_new = z (copied prev timestamp). when z_below is 1, z_new = dy.round etc

        #hier = True
#        z_tmp = dy.pick_range(fslice, self.hidden_dim*4,self.hidden_dim*4+1)
#        z_tilde = dy.logistic(z_tmp)  #original: hard sigmoid + slope annealing (a)
#        z_new = dy.cmult(1-z_below, self.z) + dy.cmult(z_below, dy.round(z_tilde, gradient_mode="straight_through_gradient"))
        
        #hier = False
        z_tmp = dy.pick_range(fslice, self.hidden_dim*4,self.hidden_dim*4+1)
        z_tilde = dy.logistic(z_tmp)  #original: hard sigmoid + slope annealing (a)
        z_new = dy.round(z_tilde, gradient_mode="straight_through_gradient")  #use straight-through estimator for gradient: step fn forward, hard sigmoid backward

        #z = z_l,t-1
        #z_below = z_l-1,t

#        if self.z.value() == 1: #FLUSH
#            c_new = dy.cmult(i_t, g_t)
#            h_new = dy.cmult(o_t, dy.tanh(c_new))
#        elif z_below.value() == 0: #COPY

        # if flush removed, only copy or normal update
        # when z_below is 0, c_new and h_new are self.c and self.h. when z_below is 1, c_new, h_new = normal update
        c_new = dy.cmult((1-z_below), self.c) + dy.cmult(z_below, (dy.cmult(f_t, self.c) + dy.cmult(i_t, g_t)))
        h_new = dy.cmult((1-z_below), self.h) + dy.cmult(z_below, dy.cmult(o_t, dy.tanh(c_new)))
        
#        if z_below.value() == 0: #COPY
#            c_new = self.c
#            h_new = self.h
#        else: #UPDATE
#            c_new = dy.cmult(f_t, self.c) + dy.cmult(i_t, g_t)
#            h_new = dy.cmult(o_t, dy.tanh(c_new))
        
        self.c = c_new
        self.h = h_new
        self.z = z_new

        return h_new, z_new


class HM_LSTMTransducer(transducers.SeqTransducer, Serializable):
    """
    hard-coded to three layers at the moment
    """
    yaml_tag = '!HM_LSTMTransducer'
    @register_xnmt_handler
    @serializable_init
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 hier=False,
                 a=1,
                 bottom_layer=None,
                 mid_layer=None,
                 top_layer=None):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.a = a  #for slope annealing
        self.bottom_layer = self.add_serializable_component("bottom_layer", bottom_layer,
                                                            lambda: HMLSTMCell(input_dim=input_dim,
                                                                               hidden_dim=hidden_dim,
                                                                               above_dim=hidden_dim,
                                                                               a=a,
                                                                               hier=hier,
                                                                               last_layer=False))
        self.mid_layer    = self.add_serializable_component("mid_layer", mid_layer,
                                                            lambda: HMLSTMCell(input_dim=hidden_dim,
                                                                               hidden_dim=hidden_dim,
                                                                               above_dim=hidden_dim,
                                                                               a=a,
                                                                               hier=hier,
                                                                               last_layer=False))
        self.top_layer    = self.add_serializable_component("top_layer", top_layer,
                                                            lambda: HMLSTMCell(input_dim=hidden_dim,
                                                                               hidden_dim=hidden_dim,
                                                                               above_dim=None,
                                                                               a=a,
                                                                               hier=hier,
                                                                               last_layer=True))
        self.modules = [self.bottom_layer, self.mid_layer, self.top_layer]

        

    @handle_xnmt_event
    def on_start_sent(self, *args, **kwargs):
        self._final_states = None

    def get_final_states(self):
        assert self._final_states is not None, "transduce() must be invoked before get_final_states()"
        return self._final_states
    

    def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':          
        batch_size = xs[0][0].dim()[1]
        h_bot = []
        h_mid = []
        h_top = []
        z_bot = []
        z_mid = []
        z_top = []

        self.top_layer.h = None
        self.top_layer.c = None
        self.top_layer.z = None
        self.mid_layer.h = None
        self.mid_layer.c = None
        self.mid_layer.z = None
        self.bottom_layer.h = None
        self.bottom_layer.c = None
        self.bottom_layer.z = None

        #?? checkme. want to init z to ones? (cherry paper)
        z_one = dy.ones(1, batch_size=batch_size)
        h_bot.append(dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)) #indices for timesteps are +1
        h_mid.append(dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size))
        h_top.append(dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size))
        
        for i, x_t in enumerate(xs):
            h_t_bot, z_t_bot = self.bottom_layer.transduce(h_below=x_t, h_above=h_mid[i], z_below=z_one) #uses h_t_top from layer above@previous time step, h_t_bot and z_t_bot from previous time step (saved in hmlstmcell)
            h_t_mid, z_t_mid = self.mid_layer.transduce(h_below=h_t_bot, h_above=h_top[i], z_below=z_t_bot) #uses h_t_top from layer above@previous time step, h_t_bot and z_t_bot from previous time step (saved in hmlstmcell)
            h_t_top, z_t_top = self.top_layer.transduce(h_below=h_t_mid, h_above=None, z_below=z_t_mid) #uses z_t_bot and h_t_bot from previous layer call, h_t_top and z_t_top from previous time step (saved in hmlstmcell)
            
            h_bot.append(h_t_bot)
            z_bot.append(z_t_bot)
            h_mid.append(h_t_mid)
            z_mid.append(z_t_mid)
            h_top.append(h_t_top)
            z_top.append(z_t_top)

#        #gated output module
#
#        #sigmoid
#        W_layer = dy.parameters(dim=(len(self.modules), hidden_dim)) #needs to be moved to init? num layers by hidden_dim
#        h_cat   = dy.transpose(dy.concatenate([h_bot, h_mid, h_top]))
#        dotted  = dy.dot_product(e1, e2)
#        gates   = dy.logistic(dotted)
#        #relu
#        
#        om = dy.relu()

        #final state is last hidden state from top layer
        self._final_states = [transducers.FinalTransducerState(h_top[-1])]
        fin_xs = expression_seqs.ExpressionSequence(expr_list=h_top[1:])
        return fin_xs #removes the init zeros to make it same length as seq
