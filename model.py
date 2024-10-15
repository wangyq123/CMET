import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder
import BertTextEncoder

__all__ = ['CMET']

class CMET(nn.Module):
    def __init__(self, args):
        super(CMET, self).__init__()
        # text subnets
        self.args = args
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        # audio-vision subnets
        text_in, audio_in, video_in = args.feature_dims

        self.tnt1 = TNT(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
        self.tnt2 = TNT(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
        self.tnt3 = TNT(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)

        self.SAl = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)
        self.SAa = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)
        self.SAv = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)

        self.SAls = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)
        self.SAas = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)
        self.SAvs = TransformerEncoder(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads,
            layers=2, attn_dropout=args.attn_dropout, relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout,
            attn_mask=True)

        self.audio_model = AuViSubNet(
            audio_in, 
            args.a_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        
        self.video_model = AuViSubNet(
            video_in, 
            args.v_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        
        self.proj_l = nn.Conv1d(text_in, args.dst_feature_dims, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(3 * args.dst_feature_dims, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.dst_feature_dims, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.dst_feature_dims, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.dst_feature_dims, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    
    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()

        text = self.text_model(text)

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        text = self.proj_l(text.transpose(1,2)) 
        
        proj_x_a = audio.permute(2, 0, 1)
        proj_x_v = video.permute(2, 0, 1)
        proj_x_l = text.permute(2, 0, 1) 
       
        text_h = self.SAls(proj_x_l)[-1]
        audio_h = self.SAas(proj_x_a)[-1]
        video_h = self.SAvs(proj_x_v)[-1] 

        fl = self.tnt1(proj_x_l, proj_x_a, proj_x_v)
        fa = self.tnt2(proj_x_a, proj_x_l, proj_x_v)
        fv = self.tnt3(proj_x_v, proj_x_a, proj_x_l)

        h_ls = self.SAl(fl)
        h_as = self.SAa(fa)
        h_vs = self.SAv(fv)

        last_h_l = h_ls[-1]
        last_h_a = h_as[-1]
        last_h_v = h_vs[-1]

        # fusion
        fusion_h = torch.cat([last_h_l, last_h_a, last_h_v], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h) 
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)

        # # text
        text_h = self.post_text_dropout(text_h)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio_h)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video_h)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)


        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h, 
            'Feature_a': audio_h,
            'Feature_v': video_h, 
            'Feature_f': fusion_h, 
        }
        return res


class TNT(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout) -> None:
        super().__init__()

        self.lower_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        self.upper_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        self.SA = TransformerEncoder(
            embed_dim=embed_dim*2,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        # 门控机制：线性变换和 sigmoid
        self.gate_fab = TSG(embed_dim, embed_dim) # seq, dim
        self.gate_fac = TSG(embed_dim, embed_dim)
        self.gate_h = TSG(embed_dim, embed_dim)

        self.fc = nn.Linear(embed_dim*2, embed_dim)
    
    def forward(self, fa, fb, fc): 
        fab = self.lower_mha(fa, fb, fb)
        fac = self.upper_mha(fa, fc, fc) 

        fab = self.gate_fab(fab)
        fac = self.gate_fac(fac)

        f1 = torch.cat([fab, fac], dim=2)  
        f = self.SA(f1, f1, f1)
        h = self.fc(f)
        h = self.gate_h(h)

        return h + fa

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, conv1d_kernel_size, dst_feature_dims, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.conv = nn.Conv1d(hidden_size, dst_feature_dims, kernel_size=conv1d_kernel_size, bias=False)
        

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size) 
        '''
        h, _ = self.rnn(x) 
        h = self.conv(h.transpose(1,2))
        return h


class TSG(nn.Module):

    def __init__(self, seq_len, dim, epsilon=1e-5):
        super(TSG, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, seq_len, 1))  # [1, seq_len, 1]
        self.gamma = nn.Parameter(torch.zeros(1, seq_len, 1))  # [1, seq_len, 1]
        self.beta = nn.Parameter(torch.zeros(1, seq_len, 1))   # [1, seq_len, 1]
        
        self.epsilon = epsilon

    def forward(self, x):

        x = x.permute(1, 0, 2)  
        embedding = (x.pow(2).sum(dim=2, keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=0, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        x = x * gate  
        x = x.permute(1, 0, 2)  

        return x  





