import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model import SimpleAttention, MatchingAttention, Attention


class CommonsenseRNNCell(nn.Module):
    def __init__(self,
                 D_m,
                 D_s,
                 D_g,
                 D_p,
                 D_r,
                 D_i,
                 D_e,
                 listener_state=False,
                 context_attention='simple',
                 D_a=100,
                 D_ca=100,
                 dropout=0.5,
                 emo_gru=True):
        super(CommonsenseRNNCell, self).__init__()
        self.D_m = D_m  # input(x) utterance
        self.D_s = D_s  #
        self.D_g = D_g  # attention a(t)
        self.D_p = D_p  # q(s)
        self.D_r = D_r  # r(s)
        self.D_i = D_i  # i(s)
        self.D_e = D_e  # y_hat or e(t)

        # print ('dmsg', D_m, D_s, D_g)
        self.g_cell = nn.GRUCell(D_m + D_p + D_r, D_g)  # GRU(C)=(X+q(s))
        self.p_cell = nn.GRUCell(
            D_s + D_g, D_p)  # GRU(Q)= (effect on speaker EScs(ut)+a(t))
        self.r_cell = nn.GRUCell(
            D_m + D_s + D_g,
            D_r)  # GRU(R)=(X+reaction of speaker RScs(ut)+a(t))
        self.i_cell = nn.GRUCell(
            D_s + D_p, D_i
        )  # GRU(I)=(intent of speaker vector IScs(ut) +q(s))     Intent State
        self.e_cell = nn.GRUCell(D_m + D_p + D_r + D_i,
                                 D_e)  # GRU(E)=(X+q(s)+r(s)+i(s))

        self.emo_gru = emo_gru
        self.listener_state = listener_state
        if listener_state:
            self.pl_cell = nn.GRUCell(
                D_s + D_g, D_p)  # GRU(Q)= (effect on listeners ELcs(ut)+a(t))
            self.rl_cell = nn.GRUCell(
                D_m + D_s + D_g,
                D_r)  # GRU(R)=(X+ reaction of listeners RLcs(ut)+a(t))

        self.dropout = nn.Dropout(dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g,D_ca)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a,D_ca)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, x1, x2, x3, o1, o2, qmask, g_hist, q0, r0, i0, e0):
        """
        U -> batch, D_m
        x1, x2, x3, o1, o2 -> batch, D_m
        x1 -> effect on self; x2 -> reaction of self; x3 -> intent of self
        o1 -> effect on others; o2 -> reaction of others
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)

        ## global state ##
        g_ = self.g_cell(
            torch.cat([U, q0_sel, r0_sel], dim=1),
            torch.zeros(U.size()[0], self.D_g).type(U.type())
            if g_hist.size()[0] == 0 else g_hist[-1])
        # g_ = self.dropout(g_)

        ## context ##
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
        else:
            c_ = self.attention(g_hist, U)

        ## external state ##
        U_r_c_ = torch.cat([U, x2, c_],
                           dim=1).unsqueeze(1).expand(-1,
                                                      qmask.size()[1], -1)
        # print ('urc', U_r_c_.size())
        # print ('u x2, c', U.size(), x2.size(), c_.size())
        rs_ = self.r_cell(
            U_r_c_.contiguous().view(-1, self.D_m + self.D_s + self.D_g),
            r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        # rs_ = self.dropout(rs_)

        ## internal state ##
        es_c_ = torch.cat([x1, c_],
                          dim=1).unsqueeze(1).expand(-1,
                                                     qmask.size()[1], -1)
        qs_ = self.p_cell(es_c_.contiguous().view(-1, self.D_s + self.D_g),
                          q0.view(-1, self.D_p)).view(U.size()[0], -1,
                                                      self.D_p)
        # qs_ = self.dropout(qs_)

        if self.listener_state:
            ## listener external state ##
            U_ = U.unsqueeze(1).expand(-1,
                                       qmask.size()[1],
                                       -1).contiguous().view(-1, self.D_m)
            er_ = o2.unsqueeze(1).expand(-1,
                                         qmask.size()[1],
                                         -1).contiguous().view(-1, self.D_s)
            ss_ = self._select_parties(rs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_r)
            U_er_ss_ = torch.cat([U_, er_, ss_], 1)
            rl_ = self.rl_cell(U_er_ss_,
                               r0.view(-1,
                                       self.D_r)).view(U.size()[0], -1,
                                                       self.D_r)
            # rl_ = self.dropout(rl_)

            ## listener internal state ##
            es_ = o1.unsqueeze(1).expand(-1,
                                         qmask.size()[1],
                                         -1).contiguous().view(-1, self.D_s)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            es_ss_ = torch.cat([es_, ss_], 1)
            ql_ = self.pl_cell(es_ss_,
                               q0.view(-1,
                                       self.D_p)).view(U.size()[0], -1,
                                                       self.D_p)
            # ql_ = self.dropout(ql_)

        else:
            rl_ = r0
            ql_ = q0

        qmask_ = qmask.unsqueeze(2)
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        r_ = rl_ * (1 - qmask_) + rs_ * qmask_

        ## intent ##
        i_q_ = torch.cat([x3, self._select_parties(q_, qm_idx)],
                         dim=1).unsqueeze(1).expand(-1,
                                                    qmask.size()[1], -1)
        is_ = self.i_cell(i_q_.contiguous().view(-1, self.D_s + self.D_p),
                          i0.view(-1, self.D_i)).view(U.size()[0], -1,
                                                      self.D_i)
        # is_ = self.dropout(is_)
        il_ = i0
        i_ = il_ * (1 - qmask_) + is_ * qmask_

        ## emotion ##
        es_ = torch.cat([
            U,
            self._select_parties(q_, qm_idx),
            self._select_parties(r_, qm_idx),
            self._select_parties(i_, qm_idx)
        ],
                        dim=1)
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0

        if self.emo_gru:
            e_ = self.e_cell(es_, e0)
        else:
            e_ = es_

        # e_ = self.dropout(e_)
        g_ = self.dropout1(g_)
        q_ = self.dropout2(q_)
        r_ = self.dropout3(r_)
        i_ = self.dropout4(i_)
        e_ = self.dropout5(e_)

        return g_, q_, r_, i_, e_


class CommonsenseRNN(nn.Module):
    def __init__(self,
                 D_m,
                 D_s,
                 D_g,
                 D_p,
                 D_r,
                 D_i,
                 D_e,
                 listener_state=False,
                 context_attention='simple',
                 D_a=100,
                 D_ca=100,
                 dropout=0.5,
                 emo_gru=True):
        super(CommonsenseRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_r = D_r
        self.D_i = D_i
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = CommonsenseRNNCell(D_m, D_s, D_g, D_p, D_r, D_i,
                                                D_e, listener_state,
                                                context_attention, D_a,D_ca,
                                                dropout, emo_gru)

    def forward(self, U, x1, x2, x3, o1, o2, qmask):
        """
        U -> seq_len, batch, D_m
        x1, x2, x3, o1, o2 -> seq_len, batch, D_s
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1],
                         qmask.size()[2],
                         self.D_p).type(U.type())  # batch, party, D_p
        r_ = torch.zeros(qmask.size()[1],
                         qmask.size()[2],
                         self.D_r).type(U.type())  # batch, party, D_r
        i_ = torch.zeros(qmask.size()[1],
                         qmask.size()[2],
                         self.D_i).type(U.type())  # batch, party, D_i

        e_ = torch.zeros(0).type(U.type())  # batch, D_e
        e = e_

        for u_, x1_, x2_, x3_, o1_, o2_, qmask_ in zip(U, x1, x2, x3, o1, o2,
                                                       qmask):
            g_, q_, r_, i_, e_ = self.dialogue_cell(
                u_, x1_, x2_, x3_, o1_, o2_, qmask_, g_hist, q_, r_, i_, e_)

            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)

        return e  # seq_len, batch, D_e


class CommonsenseGRUModel(
        nn.Module):  # CommonsenseGRUModel inherit from nn.Module class
    def __init__(self,
                 D_m,
                 D_s,
                 D_g,
                 D_p,
                 D_r,
                 D_i,
                 D_e,
                 D_h,
                 D_a=100,
                 D_ca= 100,
                 n_classes=7,
                 listener_state=False,
                 context_attention='simple',
                 recurrent_dropout=0.5,
                 dropout=0.1,
                 emo_gru=True,
                 norm=0):

        super(
            CommonsenseGRUModel,
            self).__init__()  # Call init from supr class in addition to itself

        self.norm_strategy = norm
        self.input_layer = nn.Linear(D_m, D_h)

        self.r_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

        self.normallayer = nn.LayerNorm(D_m, elementwise_affine=True)
        self.normalbatch = nn.BatchNorm1d(D_m, affine=True)

        self.dropout = nn.Dropout(dropout)
        self.dropout_recurrent = nn.Dropout(recurrent_dropout)
        self.forward_RNN = CommonsenseRNN(D_h, D_s, D_g, D_p, D_r, D_i, D_e,
                                       listener_state, context_attention, D_a,D_ca,
                                       recurrent_dropout, emo_gru)
        self.backward_RNN = CommonsenseRNN(D_h, D_s, D_g, D_p, D_r, D_i, D_e,
                                       listener_state, context_attention, D_a,D_ca,
                                       recurrent_dropout, emo_gru)
        self.matchatt = MatchingAttention(2 * D_e,
                                          2 * (D_g + D_p + D_r),
                                          att_type='general2',
                                          attention_dim=450)
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self,
                r,
                x1,
                x2,
                x3,
                o1,
                o2,
                qmask,
                umask,
                att2=False,
                return_hidden=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        seq_len, batch, feature_dim = r.size()

        if self.norm_strategy == 1:
            r = self.normallayer(r.transpose(0,
                                          1).reshape(-1, feature_dim)).reshape(
                                              -1, seq_len,
                                              feature_dim).transpose(1, 0)
        elif self.norm_strategy == 2:
            norm2 = nn.LayerNorm((seq_len, feature_dim),
                                 elementwise_affine=False)
            r = norm2(r.transpose(0, 1)).transpose(0, 1)
        elif self.norm_strategy == 3:
            r = self.normalbatch(r.transpose(0,
                                          1).reshape(-1, feature_dim)).reshape(
                                              -1, seq_len,
                                              feature_dim).transpose(1, 0)
        r = self.input_layer(r)
        emotions_f = self.forward_RNN(r, x1, x2, x3, o1, o2, qmask)
        rev_r = self._reverse_seq(r, umask)
        rev_x1 = self._reverse_seq(x1, umask)
        rev_x2 = self._reverse_seq(x2, umask)
        rev_x3 = self._reverse_seq(x3, umask)
        rev_o1 = self._reverse_seq(o1, umask)
        rev_o2 = self._reverse_seq(o2, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b = self.backward_RNN(rev_r, rev_x1, rev_x2, rev_x3,
                                            rev_o1, rev_o2, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        emotions = self.dropout_recurrent(emotions)
        if att2:
            att_emotions = []
            for t in emotions:
                att_em = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        if return_hidden:
            return hidden,  emotions
        return log_prob
