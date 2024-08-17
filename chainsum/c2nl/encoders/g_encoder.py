import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GCNConv, GATConv
from c2nl.utils.misc import sequence_mask

class TransformerConvLayer(nn.Module):
    def __init__(self,
                 d_model,
                 heads,
                 dropout):
        super(TransformerConvLayer, self).__init__()

        self.gnn_type = 0
        self.gnn = TransformerConv((d_model, d_model),
                                   d_model,
                                   heads=heads,
                                   concat=False,
                                   beta=True,
                                   dropout=dropout)
        if self.gnn_type == 1:
            self.gnn = GCNConv(d_model, d_model)
        else:
            self.gnn = GATConv(d_model, d_model, heads=heads, concat=False, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        out = self.gnn(x, edge_index)
        out = self.relu(self.layer_norm(self.dropout(out))) + x
        return out

class TransformerConvEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 dropout=0.2,
                 c_layer=None):
        super(TransformerConvEncoder, self).__init__()

        self.num_layers = num_layers
        assert c_layer is not None and len(c_layer) == num_layers
        self.layer_attn = c_layer
        self.layer = nn.ModuleList(
            [TransformerConvLayer(d_model,
                                  heads,
                                  dropout)
             for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.attn_use_method = -1
        # =-1: attn first; =0: parallel

    def count_parameters(self):
        params = list(self.layer.parameters())
        if self.attn_use_method != 0:
            params += list(self.layer_attn.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, input, input_len, edges, edge_num):
        batch_size = input.size(0)
        out = input
        layer_outputs = []      # the output of each layer
        mask = ~sequence_mask(input_len, out.shape[1]).unsqueeze(1)

        for i in range(self.num_layers):
            if self.attn_use_method == -1:
                out, _ = self.layer_attn[i](out, mask)

            tmp_out = out.clone()
            # since batch training is not supported for graph:
            for j_batch in range(batch_size):
                x = out[j_batch][:input_len[j_batch]]
                edge_index = edges[j_batch][:, :edge_num[j_batch]]
                x_out = self.layer[i](x, edge_index)
                tmp_out[j_batch, :input_len[j_batch]] = x_out
            out = tmp_out

            if self.attn_use_method == 1:
                out, _ = self.layer_attn[i](out, mask)

            layer_outputs.append(out)
        return layer_outputs


class G_Encoder(nn.Module):
    def __init__(self, args, input_size, c_encoder):
        super(G_Encoder, self).__init__()

        self.dual_enc = False
        self.dual_stack = False
        layer_attn = c_encoder.transformer.layer
        self.transformer = TransformerConvEncoder(num_layers=args.nlayers,
                                                  d_model=input_size,
                                                  heads=args.num_head,
                                                  dropout=args.trans_drop,
                                                  c_layer=layer_attn)
        if self.dual_enc:
            self.c_encoder = c_encoder
            self.transformer.attn_use_method = 0
        else:
            self.dual_stack = True
        # For convenience, we only consider cases where use_all_enc_layers is False
        assert not args.use_all_enc_layers
        # dual_enc, dual_stack, attn_use_method:
        # 0, -, -1: (ag)^l
        # 1, 0, -: (a^l)+(g^l)
        # 1, 1, -: (a^l)(g^l)

    def count_parameters(self):
        num = self.transformer.count_parameters()
        if self.dual_enc:
            num += self.c_encoder.count_parameters()
        return num

    def forward(self, input, input_len, edges, edge_num):
        input_copy = input.clone()
        input_len_copy = input_len.clone()
        if self.dual_enc:
            c_output, _ = self.c_encoder(input, input_len)
            if self.dual_stack:
                input_copy = c_output
        output = self.transformer(input_copy, input_len_copy, edges, edge_num)
        # here we only need the output from the last layer
        output = output[-1]
        if self.dual_enc and (not self.dual_stack):
            output = output + c_output

        return output
