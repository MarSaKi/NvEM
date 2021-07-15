
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

class ObjEncoder(nn.Module):
    ''' Encodes object labels using GloVe. '''

    def __init__(self, vocab_size, embedding_size, glove_matrix):
        super(ObjEncoder, self).__init__()

        padding_idx = 100
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.embedding.weight.data[...] = torch.from_numpy(glove_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        return embeds

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        if glove is not None:
            print('Using glove word embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, weighted_context, attn
        else:
            return weighted_context, attn

class ScaledSoftDotAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, output_dim):
        super(ScaledSoftDotAttention, self).__init__()
        self.scale = 1 / (output_dim**0.5)
        self.linear_q = nn.Linear(q_dim, output_dim, bias=False)
        self.linear_k = nn.Linear(k_dim, output_dim, bias=False)
        self.linear_v = nn.Sequential(nn.Linear(v_dim, output_dim), nn.Tanh())
    
    def forward(self, q_in, k_in, v_in, mask=None):
        '''
        q = B x L x D
        k = B x L x N x D
        v = B x L x N x D
        mask = B x L x N
        '''
        q = self.linear_q(q_in)
        k = self.linear_k(k_in)
        v = self.linear_v(v_in)
        attn = torch.matmul(k, q.unsqueeze(3)).squeeze(3) * self.scale
        if mask is not None:
            attn.masked_fill_(mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        v_out = torch.matmul(v.permute(0,1,3,2), attn.unsqueeze(3)).squeeze(3)

        return v_out

class ASODecoderLSTM(nn.Module):
    def __init__(self, action_embed_size, hidden_size, dropout_ratio):
        super(ASODecoderLSTM, self).__init__()
        self.action_embed_size = action_embed_size
        self.hidden_size = hidden_size
        self.action_embedding = nn.Sequential(nn.Linear(args.angle_feat_size, action_embed_size), nn.Tanh())
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.feat_att_layer = SoftDotAttention(hidden_size, args.visual_feat_size+args.angle_feat_size)
        self.lstm = nn.LSTMCell(action_embed_size+args.visual_feat_size+args.angle_feat_size, hidden_size)

        self.action_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.subject_att_layer = SoftDotAttention(hidden_size, hidden_size)
        self.object_att_layer = SoftDotAttention(hidden_size, hidden_size)

        self.fuse_a = nn.Linear(hidden_size, 1)
        self.fuse_s = nn.Linear(hidden_size, 1)
        self.fuse_o = nn.Linear(hidden_size, 1)

        self.value_action = nn.Sequential(nn.Linear(args.angle_feat_size, hidden_size), nn.Tanh())
        self.subject_att = ScaledSoftDotAttention(args.angle_feat_size, args.angle_feat_size, args.visual_feat_size, hidden_size)    
        self.object_att = ScaledSoftDotAttention(hidden_size, args.glove_dim+args.angle_feat_size, args.glove_dim+args.angle_feat_size, hidden_size)

        #cand attention layer
        self.cand_att_a = SoftDotAttention(hidden_size, hidden_size)
        self.cand_att_s = SoftDotAttention(hidden_size, hidden_size)
        self.cand_att_o = SoftDotAttention(hidden_size, hidden_size)

    def forward(self, action, feature,
                cand_visual_feat, cand_angle_feat, cand_obj_feat,
                near_visual_mask, near_visual_feat, near_angle_feat,
                near_obj_mask, near_obj_feat, near_edge_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        action_embeds = self.action_embedding(action)
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])
            cand_visual_feat = self.drop_env(cand_visual_feat)
            near_visual_feat = self.drop_env(near_visual_feat)
        cand_obj_feat = self.drop_env(cand_obj_feat)
        near_obj_feat = self.drop_env(near_obj_feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)
        concat_input = torch.cat((action_embeds, attn_feat), dim=-1)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))
        h_1_drop = self.drop(h_1)

        h_a, u_a, _ = self.action_att_layer(h_1_drop, ctx, ctx_mask)
        h_s, u_s, _ = self.subject_att_layer(h_1_drop, ctx, ctx_mask)
        h_o, u_o, _ = self.object_att_layer(h_1_drop, ctx, ctx_mask)
        h_a_drop, u_a_drop = self.drop(h_a), self.drop(u_a)
        h_s_drop, u_s_drop = self.drop(h_s), self.drop(u_s)
        h_o_drop, u_o_drop = self.drop(h_o), self.drop(u_o)

        fusion_weight = torch.cat([self.fuse_a(u_a_drop), self.fuse_s(u_s_drop), self.fuse_o(u_o_drop)], dim=-1)
        fusion_weight = F.softmax(fusion_weight, dim=-1)

        B, L = near_visual_mask.shape[0], near_visual_mask.shape[1]
        #action 
        v_action = self.value_action(cand_angle_feat)

        #subject
        v_subject = self.subject_att(cand_angle_feat, near_angle_feat, near_visual_feat, near_visual_mask)
        v_subject = self.drop(v_subject)

        #object
        near_obj = torch.cat([near_obj_feat, near_edge_feat.unsqueeze(3).expand(-1,-1,-1,args.top_N_obj,-1)], dim=-1)
        near_obj = near_obj.view(B, L, 4*args.top_N_obj, -1)
        near_obj_mask = near_obj_mask.unsqueeze(3).expand(-1,-1,-1,args.top_N_obj).contiguous().view(B, L, 4*args.top_N_obj)
        v_object = self.object_att(u_o_drop.unsqueeze(1).expand(-1,L,-1), near_obj, near_obj, near_obj_mask)
        v_object = self.drop(v_object)

        _, logit_a = self.cand_att_a(h_a_drop, v_action, output_tilde=False, output_prob=False)
        _, logit_s = self.cand_att_s(h_s_drop, v_subject, output_tilde=False, output_prob=False)
        _, logit_o = self.cand_att_o(h_o_drop, v_object, output_tilde=False, output_prob=False)
        logit = torch.cat([logit_a.unsqueeze(2), logit_s.unsqueeze(2), logit_o.unsqueeze(2)], dim=-1)
        logit = torch.matmul(logit, fusion_weight.unsqueeze(2)).squeeze(2)
        h_tilde = (h_a + h_s + h_o) / 3.

        return h_1, c_1, logit, h_tilde

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2048 + 128). The feature of the view
        :param feature: (batch_size, length, 36, 2048 + 128). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1