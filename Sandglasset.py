"""
author :wxj
SANDGLASSET: A LIGHT MULTI-GRANULARITY SELF-ATTENTIVE NETWORK FOR TIME-DOMAIN SPEECH SEPARATION
Tencent AI lab
"""
import torch.nn as nn
import torch
from torch.autograd import Variable
import math
from torch.nn.modules.activation import MultiheadAttention

"""
first edition:2021.6.24
author:wxj

issue source:YoungloLee
issue1:do not share the same block
issue2:control the size of parameters,we should use depth-wise convolution
"""


# ------------------------------------------------over_lap_and_add----------------------------------------
def overlap_and_add(signal, frame_step):
    """
    Author: Kaituo XU
    :param signal:
    :param frame_step:
    :return:
    """
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # print(subframe_length)
    # print(signal.shape)
    # print(outer_dimensions)
    # subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


# ---------------------------------------------Encoder------------------------------------------------------
class Encoder(nn.Module):
    """
    INPUT [B,T] B is batch,L is length of every input

    :arg channel_size ->the D-dimesional of 1D gated convolutional
    """

    def __init__(self, feature_dim=64, kernel_dim=2):
        super(Encoder, self).__init__()
        self.kernel_dim = kernel_dim
        self.feature_dim = feature_dim
        self.ConvLayer1 = nn.Conv1d(1, self.feature_dim, kernel_size=self.kernel_dim, stride=self.kernel_dim // 2,
                                    padding=0, dilation=1, groups=1, bias=False)
        self.ReLU = nn.ReLU()

    def forward(self, input):
        # in papers,author promise the result form conv has the same feature dim with input's
        input = input.unsqueeze(dim=1)  # ->[B,1,T] FOR 1D CONV

        # ReLU-gated 1D convolutional layer
        conv_out = self.ConvLayer1(input)
        # ReLu active
        out = self.ReLU(conv_out)  # non-negative [B,D,T]
        return out

    # in papers author don't mention the para normal,we can choose the parameters
    @staticmethod
    def normal_para(input):
        pass


# -------------------------------------------------segmentation-----------------------------------
class Segmentation_Module(nn.Module):
    """
    note: pad_segment,splite_feature,merge_feature come from dual path transformer
    github:https://github.com/ujscjj/DPTNet

    :args
    input [B,E,T]->E is feature dim
    """

    def __init__(self, segment_size):
        super(Segmentation_Module, self).__init__()
        self.segment_size = segment_size

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        # 50% overlap
        segment_stride = segment_size // 2

        # calculate the rest length if the inputs can not divided by stride
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            # zero pad for segment
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        # down sample and up sample will change the dim ,so we need to pad the value
        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        #  torch.cat([pad_aux, input, pad_aux(ignore)], 2)
        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        #  torch.cat([pad_aux(ignore), input, pad_aux], 2)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
        # segments [batch,dim,segment_size,segment_num]
        segments = segments.contiguous()
        # we need pad when down (up) sample
        # in this paper the basic kernel size is 4
        return segments, rest

    def forward(self, input):
        out, rest = self.split_feature(input, self.segment_size)
        # [B,N,L,K],L is segment size
        # print("split feature:{},rest:{}".format(out.shape, rest))
        return out, rest


# ---------------------------------------------------------sandglasset_block---------------------------------------
class Sandglasset_block(nn.Module):
    """
    one sandglasset_block includes 4 part:local RNN,Downsample,Global SAN,Upsampling
    suppose our input shape is [B,D,L,K] ,L is segment size

    """

    def __init__(self, feature_dim, hidden_size, num_layer,  # this line for local rnn
                 kernel_size, segment_size
                 ):
        super(Sandglasset_block, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.segment_size = segment_size
        # local_rnn
        self.local_rnn = local_RNN(feature_dim=self.feature_dim,
                                   hidden_size=self.hidden_size,
                                   num_layer=self.num_layer)

        # down sample
        self.down_sample = Down_sample(self.kernel_size, self.feature_dim)
        # Global SAN
        self.global_SAN = Global_SAN(self.feature_dim)
        # up sample
        self.up_sample = Up_sample(self.kernel_size, self.feature_dim, self.segment_size)

    def forward(self, input):
        local_rnn_out = self.local_rnn(input)
        dowm_sample_out, pad_value = self.down_sample(local_rnn_out)
        global_san_out = self.global_SAN(dowm_sample_out)
        up_sample_out = self.up_sample(global_san_out, pad_value)
        return up_sample_out


class local_RNN(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layer, Bi_LSTM=True, batch_first=True):
        super(local_RNN, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.bi_lstm = Bi_LSTM
        self.batch_first = batch_first
        self.num_layer = num_layer
        self.Lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            batch_first=True,
                            bidirectional=True)
        self.liner = nn.Linear(2 * hidden_size, feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)

    def forward(self, input):
        # intra_LSTM for local information
        # print("local_RNN input:{}".format(input.shape))
        inputs = input
        B, D, L, K = inputs.shape
        inputs = inputs.permute(0, 3, 2, 1).contiguous().view(B * K, L, -1)  # [B,K,L,D]
        inputs = inputs.view(B * K, L, -1)
        local_recurrent, _ = self.Lstm(inputs)
        liner_out = self.liner(local_recurrent)  # [B*K,L,D]
        # layer norm
        norm_out = self.norm(liner_out)
        out = norm_out.view(B, K, L, -1).permute(0, 3, 2, 1).contiguous()  # [B,D,L,K]
        out = out + input
        # print("local_RNN out:{}".format(out.shape))
        return out


# Down_sample
class Down_sample(nn.Module):
    def __init__(self, kernel_size, feature_dim):
        """
        in paper,author begin do inter-segment operation,but author use 1Dconv to do re-sample
        so we need to change the dim order

        input [B,D,L,K] L is segment size ,N is the number of blocks

        :param kernel_size: 4^b if b<=N/2 else 4^(N-b-1)
        but we do not change at here

        :param feature_dim:  the dim of input D
        """
        super(Down_sample, self).__init__()
        # in papers ,kernel size = stride
        self.kernel_size = kernel_size
        self.stride = self.kernel_size
        self.feature_dim = feature_dim
        # YoungloLee issue2:depth-wise convolution
        # modify after
        self.conv_layer = nn.Conv1d(self.feature_dim, self.feature_dim,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, groups=self.feature_dim)
        # modify before
        # self.conv_layer = nn.Conv1d(self.feature_dim, self.feature_dim,
        #                             kernel_size=self.kernel_size,
        #                             stride=self.stride)

    def forward(self, input):
        # [B,D,L,K]
        pad_value = 0
        # print("Down sample input:{}".format(input.shape))
        B, D, L, K = input.shape
        if K % self.kernel_size != 0:
            pad_value = K % self.kernel_size
        input = input.permute(0, 2, 3, 1).contiguous()  # [B,L,K,D]
        input = input.view(B * L, K, -1)  # [B*L,K,D]
        input = input.transpose(1, 2)  # [B*L,D,K]
        out = self.conv_layer(input)  # [B*L,D,M]
        # print("Down sample out:{}".format(out.shape))
        return out, pad_value


#  Self-Attentive Network
class Global_SAN(nn.Module):
    def __init__(self, feature_dim, nhead=4, dropout=True):
        """
        input [B*L,D,M] L is segment size,D is input dim,M is the length after sample
        """
        super(Global_SAN, self).__init__()
        self.attn = MultiheadAttention(feature_dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input.transpose(1, 2)  # [B*L,M,D]
        output = self.attn(input, input, input, attn_mask=None,
                           key_padding_mask=None)[0]
        out = self.norm(self.dropout(output) + input)
        out = out.transpose(1, 2)  # [B*L,D,M]
        return out


# Up_sample
class Up_sample(nn.Module):
    def __init__(self, kernel_size, feature_dim, segment_length):
        """
        reconstruct the out put from Self-Attentive Network

        input [B*L,D,M] L is segment size ,N is the number of blocks

        :param kernel_size: 4^b if b<=N/2 else 4^(N-b-1)
        but we do not change at here

        :param feature_dim:  the dim of input D
        """
        super(Up_sample, self).__init__()
        # in papers ,kernel size = stride
        self.kernel_size = kernel_size
        self.stride = self.kernel_size
        self.feature_dim = feature_dim
        self.segment_length = segment_length
        # YoungloLee issue2:depth-wise convolution
        # modify after
        self.conv_layer = nn.ConvTranspose1d(self.feature_dim, self.feature_dim,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride, groups=self.feature_dim)

    def forward(self, input, pad_value):
        # [B*L,D,M]
        # print("up sample input:".format(input.shape))
        out = self.conv_layer(input)  # [B*L,D,K]
        # reconstruct order to [B,D,L,K]
        A, D, K = out.shape
        assert A % self.segment_length == 0
        out = out.unsqueeze(dim=1).view(-1, self.segment_length, D, K)
        out = out.permute(0, 2, 1, 3)  # [B,D,L,K]
        if pad_value != 0:
            B, D, L, _ = out.shape
            pad = torch.zeros(B, D, L, pad_value)
            out = torch.cat([out, pad], dim=3)
            out = out.contiguous()
            # print("after pad:{}".format(out.shape))
        # print("up sample output:".format(out.shape))
        return out


# ----------------------------------------Estimation-----------------------------------
class Est_mask(nn.Module):
    """
    input PReLU[B,D,L,K]->2DCONV->[B,C*D,L,K]->[B,C*D,T]->Mask

    """

    def __init__(self, feature_dim, segment_size, nspk):
        super(Est_mask, self).__init__()
        self.feature_dim = feature_dim
        self.segment_size = segment_size
        self.nspk = nspk
        self.prelu = nn.PReLU()
        self.conv2d_layer = nn.Conv2d(self.feature_dim, self.feature_dim * self.nspk, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, input, rest):
        """
        :param input: [B,D,L,K]
        :return:
        """
        # print("EST_MASK:{}".format(input.shape))
        PRelu_out = self.prelu(input)
        conv2d_out = self.conv2d_layer(PRelu_out)  # [B,C*D,L,K]
        overlap_out = self.merge_feature(conv2d_out, rest)  # [B,C*D,T]
        mask = self.relu(overlap_out)  # [B,C*D,T]
        B, _, T = mask.shape
        mask = mask.view(B, self.nspk, -1, T)
        return mask

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)
        # B*nspk, N, L, K
        # the opposite operation of padding
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        # print("merge_feature1:{}".format(input.shape))
        # B*2, N, K, L*2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L
        # print("merge_feature2:{}".format(input.shape))

        # get speaker1
        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :,
                 segment_stride:]  # ignore the zero pad
        # print("input1:{}".format(input1.shape))
        # get speaker2
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]
        # print("input2:{}".format(input2.shape))
        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T


# -------------------------------------------Decoder------------------------------
class Decoder(nn.Module):
    def __init__(self, basic_signal, feature_dim):
        super(Decoder, self).__init__()
        self.basic_signal = basic_signal
        self.feature_dim = feature_dim
        self.liner = nn.Linear(self.feature_dim, self.basic_signal)

    def forward(self, mixture_w, est_mask):
        """
        :param input: [B,D,T]
        :param mask_c: [B,C,D,T]
        :return:
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, D, L]

        est_source = torch.transpose(source_w, 2, 3)  # [B, C, L, D]
        # don't be confused by the name of basic_signal,it just use for convenience
        # we want to use the function had been written before and do the galr_block for n times!
        # please set basic_signal is 2 to avoid error!
        est_source = self.liner(est_source)
        est_source = overlap_and_add(est_source, self.basic_signal // 2)  # B x C x T
        return est_source


# -------------------------------------------sandglasset--------------------------------
class Sandglasset(nn.Module):
    def __init__(self, nspk, feature_dim, hidden_size, num_layer, segment_size, basic_signal=2, num_block=6):
        super(Sandglasset, self).__init__()

        self.nspk = nspk
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.segment_size = segment_size
        self.basic_signal = basic_signal

        self.encoder = Encoder(feature_dim=self.feature_dim)
        self.segmentation = Segmentation_Module(segment_size=self.segment_size)
        # YoungloLee issue2: we should try not to share the same block
        # modify before
        # self.sandglass_block_1_6 = Sandglasset_block(feature_dim=self.feature_dim,
        #                                              hidden_size=self.hidden_size,
        #                                              num_layer=self.num_layer,
        #                                              kernel_size=4 ** 1,
        #                                              segment_size=self.segment_size)
        # self.sandglass_block_2_5 = Sandglasset_block(feature_dim=self.feature_dim,
        #                                              hidden_size=self.hidden_size,
        #                                              num_layer=self.num_layer,
        #                                              kernel_size=4 ** 2,
        #                                              segment_size=self.segment_size)
        # self.sandglass_block_3_4 = Sandglasset_block(feature_dim=self.feature_dim,
        #                                              hidden_size=self.hidden_size,
        #                                              num_layer=self.num_layer,
        #                                              kernel_size=4 ** 3,
        #                                              segment_size=self.segment_size)

        # modify after
        self.sandglass_block_1 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 1,
                                                   segment_size=self.segment_size)
        self.sandglass_block_6 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 1,
                                                   segment_size=self.segment_size)
        self.sandglass_block_2 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 2,
                                                   segment_size=self.segment_size)
        self.sandglass_block_5 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 2,
                                                   segment_size=self.segment_size)
        self.sandglass_block_3 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 3,
                                                   segment_size=self.segment_size)
        self.sandglass_block_4 = Sandglasset_block(feature_dim=self.feature_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layer=self.num_layer,
                                                   kernel_size=4 ** 3,
                                                   segment_size=self.segment_size)

        self.est_mask = Est_mask(feature_dim=self.feature_dim, segment_size=self.segment_size,
                                 nspk=self.nspk)
        self.decoder = Decoder(basic_signal=self.basic_signal, feature_dim=self.feature_dim)

        # use for changeable blocks
        # if num_block % 2 == 0:
        #     self.num_block = num_block
        # else:
        #     self.num_block = num_block + 1
        # Sandglass = []
        # # Sandglass
        # for index in range(1, self.num_block + 1):
        #     if index <= self.num_block // 2:
        #         Sandglass.append(Sandglasset_block(feature_dim=self.feature_dim,
        #                                            hidden_size=self.hidden_size,
        #                                            num_layer=self.num_layer,
        #                                            kernel_size=4 ** index))
        #     else:
        #         Sandglass.append(Sandglasset_block(feature_dim=self.feature_dim,
        #                                            hidden_size=self.hidden_size,
        #                                            num_layer=self.num_layer,
        #                                            kernel_size=4 ** (
        #                                                    self.num_block - index + 1)))
        #         # here is different from paper
        # self.Sandglass = Sandglass

    def forward(self, input):
        encoder_out = self.encoder(input)
        segment_out, rest = self.segmentation(encoder_out)
        out1 = self.sandglass_block_1(segment_out)
        # print("block_1 out:{}".format(out1.shape))
        out2 = self.sandglass_block_2(out1)
        # print("block_2 out:{}".format(out2.shape))
        out3 = self.sandglass_block_3(out2)
        # print("block_3 out:{}".format(out3.shape))
        out4 = self.sandglass_block_4(out3)
        out4 = out4 + out3
        # print("block_4 out:{}".format(out4.shape))
        out5 = self.sandglass_block_5(out4)
        out5 = out5 + out2
        # print("block_5 out:{}".format(out5.shape))
        out6 = self.sandglass_block_6(out5)
        out6 = out6 + out1
        # print("block_6 out:{}".format(out6.shape))
        est_mask = self.est_mask(out6, rest)
        est_src = self.decoder(encoder_out, est_mask)
        return est_src

    # if we want to change the number of block,we can edit this function
    def run(self, segment_out, Sandglass, num_block):
        """
        :param segment_out:
        :param Sandglass: list:[] the layer of Sandglasset_block
        :param num_block: the number of Sandglasset_blocks
        :return:
        """
        pass

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['nspk'], package['feature_dim'],
                    package['hidden_size'], package['num_layer'],
                    segment_size=package['segment_size'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'nspk': model.nspk,
            'feature_dim': model.feature_dim,
            'hidden_size': model.hidden_size,
            'num_layer': model.num_layer,
            'segment_size': model.segment_size,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == "__main__":
    sandglass = Sandglasset(feature_dim=32, nspk=2, hidden_size=20, num_layer=3,
                            segment_size=256)
    encoder = Encoder(feature_dim=64)
    input = torch.zeros(1, 70000)
    print(sandglass(input).shape)
