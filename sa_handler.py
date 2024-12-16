# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
from sklearn.decomposition import PCA
import math

T = torch.Tensor


@dataclass
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    enable_attention_sharing: bool = True,
    attention_sharing_steps_flag: bool = False,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.
    shared_score_shift: float = 0.
    only_self_level: float = 0.
    replace_attention: bool = False
    start_attention_sharing: int = -1
    stop_attention_sharing: int = 10000
    shared_attn_name: str = None
    layer_to_null: str = None
    head_to_null: int = None
    channel_to_null: int = None
    channel_to_inf: int = None
    channel_to_null_partition: float = None
    svd: bool = False
    start_svd: int = -1
    end_svd: int = 10000
    svd_flag: bool = False
    svd_coef: float = None
    svd_remain_one: int = None
    svd_null_start: int = None
    svd_null_end: int = None
    head_num: int = None
    head_layer: str = None
    
    svd_variance_value: bool = False
    svd_variance_value_biggest: bool = False
    svd_variance_value_smallest: bool = False
    svd_variance_value_threshhold: float = None
    svd_variance_key: bool = False
    svd_variance_key_biggest: bool = False
    svd_variance_key_smallest: bool = False
    svd_variance_key_threshhold: float = None
    svd_mass_value: bool = False
    svd_mass_value_threshhold: float = None
    
    svd_reference_value: bool = False
    svd_reference_value_coef: float = 0
    svd_reference_value_start: int = None
    svd_reference_value_end: int = None
    svd_reference_value_positive: bool = False 
    svd_reference_value_negative: bool = False 
    
    svd_reference_key: bool = False
    svd_reference_key_coef: float = 0
    svd_reference_key_start: int = None
    svd_reference_key_end: int = None
    svd_reference_key_positive: bool = False 
    svd_reference_key_negative: bool = False 
    
    null_channel_reference: bool = False
    null_channel_reference_partition: float = None
    score_svd: bool = False
    score_svd_start: int = None
    score_svd_end: int = None
    scores_reference_svd: bool = False
    scores_reference_svd_start: int = None
    scores_reference_svd_end: int = None
    weights_key_svd: bool = False
    weights_key_svd_start: int = None
    weights_key_svd_end: int = None
    weights_query_svd: bool = False
    weights_query_svd_start: int = None
    weights_query_svd_end: int = None
    weights_value_svd: bool = False
    weights_value_svd_start: int = None
    weights_value_svd_end: int = None
    
    query_dropout: bool = False
    key_dropout: bool = False
    value_dropout: bool = False
    
    svd_variance_threshold_smallest: float = None



def expand_first(feat: T, scale=1.,) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def svd_cumsum(matrix, threshold, dim=-1, type='variance', biggest=True):
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    
    means = s.mean(dim=dim, keepdim=True)
    if type == 'variance':
        coefs = (
            torch.pow(
                (s - means), 2) / 
                ((s.shape[dim]-1) * s.var(dim=-1, keepdim=True))
        )
    
    elif type == 'mass':
        coefs = s / s.sum(dim=dim, keepdim=True)
        
    else:
        raise NotImplementedError('svd type should have "variance" or "mass" value')
    
    
    cumsum = coefs.cumsum(dim=dim) if biggest else 1 - coefs.cumsum(dim=dim)
    
    s[cumsum <= threshold] = 0
    # нахожу индекс первого ненулевого элемента и его cumsum вклада 
    idx = torch.argmax(s, dim=dim, keepdim=True) if biggest else torch.argmin(s, dim=dim, keepdim=True)
    bound = torch.gather(cumsum, dim=dim, index=idx)
    
    # надо найти prev index, чтобы узнать, как сильно мы обнулили
    prev_idx = idx.detach().clone()
    prev_idx[prev_idx != 0] -= 1
    prev_value = torch.gather(cumsum, dim=dim, index=prev_idx)
    
    prev_value[idx == prev_idx] = 0 if biggest else 1
    
    if not biggest:
        torch.utils.swap_tensors(prev_value, bound)
        
    # это то, сколько нам не хватило / вклад конкретно нужного элемента в массу
    k = (threshold - prev_value) / (bound - prev_value) 
    # => получили долю от чиселки, которую мы хотим обнулить, k < 1
    
    if type == 'variance':
        s.scatter_reduce_(dim, idx, -means, reduce='sum')
        s.scatter_reduce_(dim, idx, torch.sqrt(k), reduce='prod')
        s.scatter_reduce_(dim, idx, means, reduce='sum')
        
    elif type == 'mass':
        s.scatter_reduce_(-1, idx, 1 - k, reduce='prod')
    
    return u @ torch.diag_embed(s) @ v



def svd(matrix, start=0, end=None, dim=-1):
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    
    end = s.shape[-1] if end is None else end
    
    indices = torch.arange(start, end, device=s.device)
    full_indices = [slice(None)] * s.ndim
    full_indices[dim] = indices
    s[full_indices] = 0

    return u @ torch.diag_embed(s) @ v


def svd_weights(layer, start=0, end=None):
    bias = (layer.bias is not None)
            
    new_layer = nn.Linear(
        in_features=layer.in_features, 
        out_features=layer.out_features, 
        bias=bias, device=layer.weight.data.device
    )

    with torch.no_grad():
        new_layer.weight.data = svd(layer.weight.data, start=start, end=end, dim=-1)
        if bias:
            new_layer.bias.data = layer.bias.data.clone()
        
        new_layer.eval()
        
    return new_layer

def dropout(matrix, coef=2, dim=-2):
    idx = torch.randperm(matrix.shape[dim])[:matrix.shape[dim] // coef]
    full_indices = [slice(None)] * matrix.ndim
    full_indices[dim] = idx
    
    matrix[full_indices] = torch.mean(matrix, dim=dim, keepdim=True)
    return matrix


class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor2_0()

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)


class SharedAttentionProcessor(DefaultAttentionProcessor):

    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):
        # print('in shared call')
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                  
             

        # key = attn.to_k(hidden_states)
        # query = attn.to_q(hidden_states)
        # value = attn.to_v(hidden_states)
        
        if self.weights_query_svd:
            bias = (attn.to_q.bias is not None)
            
            new_layer = nn.Linear(
                in_features=attn.to_q.in_features, 
                out_features=attn.to_q.out_features, 
                bias=bias, device=hidden_states.device
            )
        
            with torch.no_grad():
                u, s, v = torch.linalg.svd(attn.to_q.weight.data, full_matrices=False)
                if self.weights_query_svd_end is None:
                    self.weights_query_svd_end = s.shape[-1]
                s[..., self.weights_query_svd_start:self.weights_query_svd_end] = 0
                new_layer.weight.data = u @ torch.diag_embed(s) @ v
                
                if bias:
                    new_layer.bias.data = attn.to_q.bias.data.clone()
                
                new_layer.eval()
                query = new_layer(hidden_states)
        else:
            query = attn.to_q(hidden_states)
            if self.query_dropout:
                # replaced = nn.functional.interpolate(
                #     nn.functional.interpolate(query.transpose(-1, -2), scale_factor=0.5, mode='linear'), 
                #     scale_factor=2, mode='linear').transpose(-1, -2)
                idx = torch.randperm(query.shape[-2])[:query.shape[-2] // 4]
                # query[:, idx, :] = replaced[:, idx, :]
                query[:, idx, :] = torch.mean(query, dim=1, keepdim=True)
            
        if self.weights_key_svd:
            bias = (attn.to_k.bias is not None)
            
            new_layer = nn.Linear(
                in_features=attn.to_k.in_features, 
                out_features=attn.to_k.out_features, 
                bias=bias, device=hidden_states.device
            )
        
            with torch.no_grad():
                u, s, v = torch.linalg.svd(attn.to_k.weight.data, full_matrices=False)
                if self.weights_key_svd_end is None:
                    self.weights_key_svd_end = s.shape[-1]
                s[..., self.weights_key_svd_start:self.weights_key_svd_end] = 0
                new_layer.weight.data = u @ torch.diag_embed(s) @ v
                
                if bias:
                    new_layer.bias.data = attn.to_k.bias.data.clone()
                
                new_layer.eval()
                key = new_layer(hidden_states)
        else:
            key = attn.to_k(hidden_states)
            if self.key_dropout:
                # replaced = nn.functional.interpolate(
                #     nn.functional.interpolate(key.transpose(-1, -2), scale_factor=0.5, mode='linear'), 
                #     scale_factor=2, mode='linear').transpose(-1, -2)
                idx = torch.randperm(key.shape[-2])[:key.shape[-2] // 4]
                # key[:, idx, :] = replaced[:, idx, :]
                key[:, idx, :] = torch.mean(key, dim=1, keepdim=True)
                
            
        if self.weights_value_svd:
            bias = (attn.to_v.bias is not None)
            
            new_layer = nn.Linear(
                in_features=attn.to_v.in_features, 
                out_features=attn.to_v.out_features, 
                bias=bias, device=hidden_states.device
            )
        
            with torch.no_grad():
                u, s, v = torch.linalg.svd(attn.to_v.weight.data, full_matrices=False)
                if self.weights_value_svd_start is None:
                    self.weights_value_svd_start = s.shape[-1]
                s[..., self.weights_value_svd_start:self.weights_value_svd_end] = 0
                new_layer.weight.data = u @ torch.diag_embed(s) @ v
                
                if bias:
                    new_layer.bias.data = attn.to_v.bias.data.clone()
                
                new_layer.eval()
                value = new_layer(hidden_states)
        else:
            value = attn.to_v(hidden_states)
            if self.value_dropout:
                # replaced = nn.functional.interpolate(
                #     nn.functional.interpolate(value.transpose(-1, -2), scale_factor=0.5, mode='linear'), 
                #     scale_factor=2, mode='linear').transpose(-1, -2)
                idx = torch.randperm(value.shape[-2])[:value.shape[-2] // 4]
                # value[:, idx, :] = replaced[:, idx, :]
                value[:, idx, :] = torch.mean(value, dim=1, keepdim=True)
            
        
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if self.step >= self.start_inject:
        if self.adain_queries:
            query = adain(query)
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)
        
        # print(torch.min(value), torch.mean(value))
        
        if self.to_null:
            if self.head_to_null is not None:
                value[:, self.head_to_null, ...] = 0
                # print(self.head_to_null)
                # print('in head_to_null')
            if self.channel_to_null is not None:
                # print('in channel_to_null')
                value[..., self.channel_to_null] = 0
            if self.channel_to_null_partition is not None:
                reshape = [1] * len(value.shape)
                reshape[-1] = -1
                
                indexes = _get_switch_vec(value.shape[-1], self.channel_to_null_partition) == True
                indexes = indexes.view(reshape)
                
                to_repeat = list(value.shape)
                to_repeat[-1] = 1
                
                # print(value.shape, indexes.shape, to_repeat)    
                value[indexes.repeat(to_repeat)] = 0
                
                
        
        #     if self.svd_reference_key:
        #         # print('in svd reference')
        #         batch = key.shape[0] // 2
        #         ref = key.shape[-2] // 2
        #         u, s, v = torch.linalg.svd(key[1:batch, :, ref:, :], full_matrices=False)
        #         s[..., self.svd_reference_key_start:self.svd_reference_key_end] = 0
        #         key[1:batch, :, ref:, :] = u @ torch.diag_embed(s) @ v
                
        #         u, s, v = torch.linalg.svd(key[batch + 1:, :, ref:, :], full_matrices=False)
        #         s[..., self.svd_reference_key_start:self.svd_reference_key_end] = 0
        #         key[batch + 1:, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                
                
        
        if self.enable_attention_sharing and self.attention_sharing_steps_flag:
            key = concat_first(key, -2, scale=self.shared_score_scale)
            # print('init', value.shape)
            value = concat_first(value, -2)
            # print('then', value.shape)
            
            if self.svd_reference_value and self.svd_flag:
                # print('in svd reference')
                batch = value.shape[0] // 2
                ref = value.shape[-2] // 2
                
                if self.svd_mass_value:
                    u, s, v = torch.linalg.svd(value[..., ref:, :], full_matrices=False)
                    
                    coefs = s / s.sum(dim=-1, keepdim=True)
                    cumsum = coefs.cumsum(dim=-1)
                    
                    s[cumsum <= self.svd_mass_value_threshhold] = 0
                    # нахожу индекс первого ненулевого элемента и его cumsum вклада в абсолютную массу
                    idx = torch.argmax(s, dim=-1, keepdim=True)
                    print(idx[0, 0])
                    bound = torch.gather(cumsum, dim=-1, index=idx)
                    print(bound[0, 0])
                    
                    # надо найти prev index, чтобы узнать, как сильно мы обнулили
                    prev_idx = idx.detach().clone()
                    prev_idx[prev_idx != 0] -= 1
                    prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                    
                    prev_value[idx == prev_idx] = 0
                    # это то, сколько нам не хватило / вклад конкретно нужного элемента в массу
                    k = (self.svd_mass_value_threshhold - prev_value) / (bound - prev_value)
                    # => получили долю абсолютной массы, которую мы хотим обнулить 
                    # k < 1
                    
                    s.scatter_reduce_(-1, idx, 1 - k, reduce='prod')
                    
                    value[..., ref:, :] = u @ torch.diag_embed(s) @ v
                    value[0, :, ref:, :] = value[0, :, :ref, :]
                    value[batch, :, ref:, :] = value[batch, :, :ref, :]
                    
                    
                
                if self.svd_variance_value:
                    u, s, v = torch.linalg.svd(value[1:batch, :, ref:, :], full_matrices=False)
                    # боже прости меня грешницу за такой плохой код
                    # короче я тут для каждого значения смотрю, какую-то долю дисперсии оно "доставляет"
                    # и потом беру камсам, чтобы посмотреть, где у нас набирается порог
                    # боже я каюсь в своем грехе нечитаемого кода
                    
                    means = s.mean(dim=-1, keepdim=True)
                    cumsum = (
                        torch.pow(
                            (s - means), 2) / 
                            ((s.shape[-1]-1) * s.var(dim=-1, keepdim=True))
                    ).cumsum(dim=-1)
                    
                    if self.svd_variance_value_biggest:
                        s[cumsum <= self.svd_variance_value_threshhold] = 0
                        # нахожу индекс первого ненулевого элемента и его cumsum вклада в дисперсию
                        idx = torch.argmax(s, dim=-1, keepdim=True)
                        bound = torch.gather(cumsum, dim=-1, index=idx)
                        
                        # надо найти prev index, чтобы узнать, как сильно мы обнулили
                        prev_idx = idx.detach().clone()
                        prev_idx[prev_idx != 0] -= 1
                        prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                        
                        prev_value[idx == prev_idx] = 0
                        # это то, сколько нам не хватило / вклад конкретно нужного элемента в дисперсии
                        k = (self.svd_variance_value_threshhold - prev_value) / (bound - prev_value)
                        # k < 1
                        
                        s.scatter_reduce_(-1, idx, -means, reduce='sum')
                        s.scatter_reduce_(-1, idx, torch.sqrt(k), reduce='prod')
                        s.scatter_reduce_(-1, idx, means, reduce='sum')
                        
                    if self.svd_variance_value_smallest:
                        s[cumsum >= self.svd_variance_value_threshhold] = 0
                    
                    value[1:batch, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    
                    u, s, v = torch.linalg.svd(value[batch + 1:, :, ref:, :], full_matrices=False)
                    cumsum = (
                        torch.pow(
                            (s - means), 2) / 
                            ((s.shape[-1]-1) * s.var(dim=-1, keepdim=True))
                    ).cumsum(dim=-1)
                    
                    if self.svd_variance_value_biggest:
                        s[cumsum <= self.svd_variance_value_threshhold] = 0
                        
                        # нахожу индекс первого ненулевого элемента и его cumsum вклада в дисперсию
                        idx = torch.argmax(s, dim=-1, keepdim=True)
                        bound = torch.gather(cumsum, dim=-1, index=idx)
                        
                        # надо найти prev index, чтобы узнать, как сильно мы обнулили
                        prev_idx = idx.detach().clone()
                        prev_idx[prev_idx != 0] -= 1
                        prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                        
                        prev_value[idx == prev_idx] = 0
                        # это то, сколько нам не хватило / вклад конкретно нужного элемента в дисперсии
                        k = (self.svd_variance_value_threshhold - prev_value) / (bound - prev_value)
                        # k < 1
                        
                        s.scatter_reduce_(-1, idx, -means, reduce='sum')
                        s.scatter_reduce_(-1, idx, torch.sqrt(k), reduce='prod')
                        s.scatter_reduce_(-1, idx, means, reduce='sum')
                        
                    if self.svd_variance_value_smallest:
                        s[cumsum >= self.svd_variance_value_threshhold] = 0
                    
                    value[batch + 1:, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    
                else:
                    if self.svd_reference_value_negative:
                        u, s, v = torch.linalg.svd(value[1:batch, :, ref:, :], full_matrices=False)
                        if self.svd_reference_value_end is None:
                            self.svd_reference_value_end = s.shape[-1]
                        s[..., self.svd_reference_value_start:self.svd_reference_value_end] *= self.svd_reference_value_coef
                        value[1:batch, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    if self.svd_reference_value_positive:
                        u, s, v = torch.linalg.svd(value[batch + 1:, :, ref:, :], full_matrices=False)
                        if self.svd_reference_value_end is None:
                            self.svd_reference_value_end = s.shape[-1]
                        s[..., self.svd_reference_value_start:self.svd_reference_value_end] *= self.svd_reference_value_coef
                        value[batch + 1:, :, ref:, :] = u @ torch.diag_embed(s) @ v
                
            if self.svd_reference_key and self.svd_flag:
                # print('in svd reference')
                batch = key.shape[0] // 2
                ref = key.shape[-2] // 2
                
                if self.svd_variance_key:
                    u, s, v = torch.linalg.svd(key[1:batch, :, ref:, :], full_matrices=False)
                    means = s.mean(dim=-1, keepdim=True)
                    cumsum = (
                        torch.pow(
                            (s - means), 2) / 
                            ((s.shape[-1]-1) * s.var(dim=-1, keepdim=True))
                    ).cumsum(dim=-1)
                    
                    if self.svd_variance_key_biggest:
                        s[cumsum <= self.svd_variance_key_threshhold] = 0
                        # нахожу индекс первого ненулевого элемента и его cumsum вклада в дисперсию
                        idx = torch.argmax(s, dim=-1, keepdim=True)
                        bound = torch.gather(cumsum, dim=-1, index=idx)
                        
                        # надо найти prev index, чтобы узнать, как сильно мы обнулили
                        prev_idx = idx.detach().clone()
                        prev_idx[prev_idx != 0] -= 1
                        prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                        
                        prev_value[idx == prev_idx] = 0
                        # это то, сколько нам не хватило / вклад конкретно нужного элемента в дисперсии
                        k = (self.svd_variance_key_threshhold - prev_value) / (bound - prev_value)
                        # k < 1
                        
                        s.scatter_reduce_(-1, idx, -means, reduce='sum')
                        s.scatter_reduce_(-1, idx, torch.sqrt(k), reduce='prod')
                        s.scatter_reduce_(-1, idx, means, reduce='sum')
                        
                        
                    if self.svd_variance_key_smallest:
                        s[cumsum >= self.svd_variance_key_threshhold] = 0
                    
                    key[1:batch, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    u, s, v = torch.linalg.svd(key[batch + 1:, :, ref:, :], full_matrices=False)
                    
                    means = s.mean(dim=-1, keepdim=True)
                    cumsum = (
                        torch.pow(
                            (s - means), 2) / 
                            ((s.shape[-1]-1) * s.var(dim=-1, keepdim=True))
                    ).cumsum(dim=-1)
                    
                    if self.svd_variance_key_biggest:
                        s[cumsum <= self.svd_variance_key_threshhold] = 0
                        # нахожу индекс первого ненулевого элемента и его cumsum вклада в дисперсию
                        idx = torch.argmax(s, dim=-1, keepdim=True)
                        bound = torch.gather(cumsum, dim=-1, index=idx)
                        
                        # надо найти prev index, чтобы узнать, как сильно мы обнулили
                        prev_idx = idx.detach().clone()
                        prev_idx[prev_idx != 0] -= 1
                        prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                        
                        prev_value[idx == prev_idx] = 0
                        # это то, сколько нам не хватило / вклад конкретно нужного элемента в дисперсии
                        k = (self.svd_variance_key_threshhold - prev_value) / (bound - prev_value)
                        # k < 1
                        
                        s.scatter_reduce_(-1, idx, -means, reduce='sum')
                        s.scatter_reduce_(-1, idx, torch.sqrt(k), reduce='prod')
                        s.scatter_reduce_(-1, idx, means, reduce='sum')
                        
                    if self.svd_variance_key_smallest:
                        s[cumsum > self.svd_variance_key_threshhold] = 0
                        
                    key[batch + 1:, :, ref:, :] = u @ torch.diag_embed(s) @ v
                
                else:
                    if self.svd_reference_key_negative:
                        u, s, v = torch.linalg.svd(key[1:batch, :, ref:, :], full_matrices=False)
                        
                        if self.svd_reference_key_end is None:
                            self.svd_reference_key_end = s.shape[-1]
                        s[..., self.svd_reference_key_start:self.svd_reference_key_end] *= self.svd_reference_key_coef
                        key[1:batch, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    if self.svd_reference_key_positive:
                        u, s, v = torch.linalg.svd(key[batch + 1:, :, ref:, :], full_matrices=False)
                        if self.svd_reference_key_end is None:
                            self.svd_reference_key_end = s.shape[-1]
                        s[..., self.svd_reference_key_start:self.svd_reference_key_end] *= self.svd_reference_key_coef
                        key[batch + 1:, :, ref:, :] = u @ torch.diag_embed(s) @ v
                    
                    # u, s, v = torch.linalg.svd(key, full_matrices=False)
                    # s[..., self.svd_reference_key_start:self.svd_reference_key_end] = 0
                    # key = u @ torch.diag_embed(s) @ v
                
                
            if self.null_channel_reference and self.svd_flag:                
                indexes = _get_switch_vec(value.shape[-1], self.null_channel_reference_partition) == True

                batch = value.shape[0] // 2
                patches = value.shape[-2] // 2
                # это типа у негативного промпта для остальных картинок обнуляю каналы у референса
                value[1:batch, :, patches:, indexes] = 0
                # это у позитивного
                value[batch + 1:, :, patches:, indexes] = 0
                
            # if self.to_uniform:
            # print('sa_handler QKV', query.shape, key.shape, value.shape)
        if self.shared_score_shift != 0:
            hidden_states = self.shifted_scaled_dot_product_attention(attn, query, key, value,)
        else:
            if self.score_svd:
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                
                # print('in svd reference')
                batch = value.shape[0] // 2
                ref = value.shape[-2] // 2
                
                if self.attention_sharing_steps_flag and self.scores_reference_svd:
                    u, s, v = torch.linalg.svd(attn_weight[1:batch, ..., ref:], full_matrices=False)
                    
                    if self.scores_reference_svd_end is None:
                        self.scores_reference_svd_end = s.shape[-1]
                        
                    s[..., self.scores_reference_svd_start:self.scores_reference_svd_end] = 0
                    attn_weight[1:batch, ..., ref:] = u @ torch.diag_embed(s) @ v
                    
                    u, s, v = torch.linalg.svd(value[batch + 1:, ..., ref:], full_matrices=False)
                    s[..., self.scores_reference_svd_start:self.scores_reference_svd_end] = 0
                    attn_weight[batch + 1:, ..., ref:] = u @ torch.diag_embed(s) @ v
                    
                else:
                    u, s, v = torch.linalg.svd(attn_weight, full_matrices=False)
                    if self.score_svd_end is None:
                        self.score_svd_end = s.shape[-1]
                    s[..., self.score_svd_start:self.score_svd_end] = 0
                    attn_weight = u @ torch.diag_embed(s) @ v
                    
                attn_weight = torch.softmax(attn_weight, dim=-1)
                hidden_states = attn_weight @ value
                
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                # hidden_states = nnf.scaled_dot_product_attention(
                #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                # )


        # else:
        #     hidden_states = nnf.scaled_dot_product_attention(
        #         query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        #     )
        if self.head_num is not None:
            hidden_states = hidden_states[:, self.head_num, ...].unsqueeze(1).repeat(1, hidden_states.shape[1], 1, 1)
        # hidden_states = adain(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        if (self.svd and self.svd_flag and 
            (self.svd_null_start is not None or 
             self.svd_remain_one is not None or 
             self.svd_coef is not None or 
             self.svd_variance_threshold_smallest is not None)): 
            # боже какой ужас
            
            # print('in svd')
            u, s, v = torch.linalg.svd(hidden_states, full_matrices=False)
            # print(hidden_states.shape, s.shape)
            if self.svd_variance_threshold_smallest is not None:
                means = s.mean(dim=-1, keepdim=True)
                cumsum = (
                    torch.pow(
                        (s - means), 2) / 
                        ((s.shape[-1]-1) * s.var(dim=-1, keepdim=True))
                ).cumsum(dim=-1)
                
                cumsum = 1 - cumsum
                    
                s[cumsum <= self.svd_variance_threshold_smallest] = 0
                # нахожу индекс первого нулевого элемента
                # это как раз граница, откуда мы начали обнулять
                idx = torch.argmin(s, dim=-1, keepdim=True)
                bound = torch.gather(cumsum, dim=-1, index=idx)
                
                # prev index -- это чиселка, которая не обнуленная еще
                prev_idx = idx.detach().clone()
                prev_idx[prev_idx != 0] -= 1
                prev_value = torch.gather(cumsum, dim=-1, index=prev_idx)
                
                prev_value[idx == prev_idx] = 1
                # это то, сколько нам не хватило / вклад конкретно нужного элемента в дисперсии
                k = (self.svd_variance_threshold_smallest - bound) / (prev_value - bound)
                # k < 1
                
                s.scatter_reduce_(-1, idx, -means, reduce='sum')
                s.scatter_reduce_(-1, idx, torch.sqrt(k), reduce='prod')
                s.scatter_reduce_(-1, idx, means, reduce='sum')
            
            if self.svd_null_start is not None:
                self.svd_null_end = self.svd_null_end if self.svd_null_end is not None else s.shape[-1]
                s[..., self.svd_null_start:self.svd_null_end] = 0
            if self.svd_remain_one is not None:
                s[..., 0:self.svd_remain_one] = 0
                s[..., (self.svd_remain_one + 1):s.shape[-1]] = 0
            if self.svd_coef is not None:
                s[..., int(s.shape[0] * self.svd_coef):] = 0
                
            hidden_states = u @ torch.diag_embed(s) @ v
        
        return hidden_states

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if self.full_attention_share:
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, '(k b) n d -> k (b n) d', k=2)
            hidden_states = super().__call__(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=attention_mask, **kwargs)
            hidden_states = einops.rearrange(hidden_states, 'k (b n) d -> (k b) n d', n=n)
        else:
            hidden_states = self.shared_call(attn, hidden_states, hidden_states, attention_mask, **kwargs)

        return hidden_states
                                                                            
    def __init__(self, style_aligned_args: StyleAlignedArgs, to_null=False, head_exp=False):
    # def __init__(self, style_aligned_args: StyleAlignedArgs, to_null=False, head_to_null=None, head_exp=False):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.enable_attention_sharing = style_aligned_args.enable_attention_sharing
        self.attention_sharing_steps_flag = style_aligned_args.attention_sharing_steps_flag
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift
        self.replace_attention = style_aligned_args.replace_attention
        self.start_attention_sharing = style_aligned_args.start_attention_sharing
        self.stop_attention_sharing = style_aligned_args.stop_attention_sharing
        self.head_to_null = style_aligned_args.head_to_null
        self.channel_to_null = style_aligned_args.channel_to_null
        self.svd = style_aligned_args.svd
        self.svd_coef = style_aligned_args.svd_coef
        self.svd_null_start = style_aligned_args.svd_null_start
        self.svd_null_end = style_aligned_args.svd_null_end
        self.channel_to_null_partition = style_aligned_args.channel_to_null_partition
        self.null_channel_reference = style_aligned_args.null_channel_reference
        self.null_channel_reference_partition = style_aligned_args.null_channel_reference_partition
        self.channel_to_inf = style_aligned_args.channel_to_inf
        self.svd_remain_one = style_aligned_args.svd_remain_one
        
        self.svd_variance_value = style_aligned_args.svd_variance_value
        self.svd_variance_value_biggest = style_aligned_args.svd_variance_value_biggest
        self.svd_variance_value_smallest = style_aligned_args.svd_variance_value_smallest
        self.svd_variance_value_threshhold = style_aligned_args.svd_variance_value_threshhold
        self.svd_variance_key = style_aligned_args.svd_variance_key
        self.svd_variance_key_biggest = style_aligned_args.svd_variance_key_biggest
        self.svd_variance_key_smallest = style_aligned_args.svd_variance_key_smallest
        self.svd_variance_key_threshhold = style_aligned_args.svd_variance_key_threshhold
        
        self.svd_reference_value = style_aligned_args.svd_reference_value
        self.svd_reference_value_coef = style_aligned_args.svd_reference_value_coef
        self.svd_reference_value_start = style_aligned_args.svd_reference_value_start
        self.svd_reference_value_end = style_aligned_args.svd_reference_value_end
        self.svd_reference_value_positive = style_aligned_args.svd_reference_value_positive
        self.svd_reference_value_negative = style_aligned_args.svd_reference_value_negative
        
        self.svd_reference_key = style_aligned_args.svd_reference_key
        self.svd_reference_key_coef = style_aligned_args.svd_reference_key_coef
        self.svd_reference_key_start = style_aligned_args.svd_reference_key_start
        self.svd_reference_key_end = style_aligned_args.svd_reference_key_end
        self.svd_reference_key_positive = style_aligned_args.svd_reference_key_positive
        self.svd_reference_key_negative = style_aligned_args.svd_reference_key_negative
        
        self.svd_flag = style_aligned_args.svd_flag
        self.score_svd = style_aligned_args.score_svd
        self.score_svd_start = style_aligned_args.score_svd_start
        self.score_svd_end = style_aligned_args.score_svd_end
        
        self.scores_reference_svd = style_aligned_args.scores_reference_svd
        self.scores_reference_svd_start = style_aligned_args.scores_reference_svd_start
        self.scores_reference_svd_end = style_aligned_args.scores_reference_svd_end
        
        self.weights_key_svd = style_aligned_args.weights_key_svd
        self.weights_key_svd_start = style_aligned_args.weights_key_svd_start
        self.weights_key_svd_end = style_aligned_args.weights_key_svd_end
        
        self.weights_query_svd = style_aligned_args.weights_query_svd
        self.weights_query_svd_start = style_aligned_args.weights_query_svd_start
        self.weights_query_svd_end = style_aligned_args.weights_query_svd_end
        
        self.weights_value_svd = style_aligned_args.weights_value_svd
        self.weights_value_svd_start = style_aligned_args.weights_value_svd_start
        self.weights_value_svd_end = style_aligned_args.weights_value_svd_end
        
        self.query_dropout = style_aligned_args.query_dropout
        self.key_dropout = style_aligned_args.key_dropout
        self.value_dropout = style_aligned_args.value_dropout
        
        self.svd_variance_threshold_smallest = style_aligned_args.svd_variance_threshold_smallest
        
        self.svd_mass_value = style_aligned_args.svd_mass_value
        self.svd_mass_value_threshhold = style_aligned_args.svd_mass_value_threshhold
    
        self.to_null = to_null
        
        self.head_num = None
        if head_exp:
            self.head_num = style_aligned_args.head_num
            
            
        # self.head_to_null = head_to_null # ПОТОМ УДАЛИТЬ
        # self.head_to_null = None # ПОТОМ УДАЛИТЬ
        # self.channel_to_null = head_to_null # ПОТОМ УДАЛИТЬ


def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > .5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec


def init_attention_processors(pipeline: StableDiffusionXLPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    attn_procs = {}
    unet = pipeline.unet
    number_of_self, number_of_cross = 0, 0
    shared_names = []
    # print([name for name in unet.attn_processors.keys() if 'attn1' in name])
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1)
    else:
        only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level)
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
            number_of_self += 1
            if style_aligned_args is None or only_self_vec[i // 2]:
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                if style_aligned_args.shared_attn_name is None or name == style_aligned_args.shared_attn_name:
                    # if name == style_aligned_args.shared_attn_name:
                    # print(name)
                    # heads = (name in style_aligned_args.layer_to_null.keys())
                    # if heads:
                    #     print('TRUE hello', name)
                    attn_procs[name] = SharedAttentionProcessor(
                        style_aligned_args, 
                        (style_aligned_args.layer_to_null == name) or (style_aligned_args.layer_to_null == 'ALL'),
                        # heads, # ПОТОМ УДАЛИТЬ
                        # None if heads is False else style_aligned_args.layer_to_null[name], # ПОТОМ УДАЛИТЬ
                        (style_aligned_args.head_layer == name)
                    )
                    shared_names.append('.'.join(name.split('.')[:-1]))
                else:
                    attn_procs[name] = DefaultAttentionProcessor()
        else:
            number_of_cross += 1
            attn_procs[name] = DefaultAttentionProcessor()
    unet.set_attn_processor(attn_procs)
    return shared_names


def register_shared_norm(pipeline: StableDiffusionXLPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True, ):
    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.unet, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in
                                                                               norm_layers['layer']]


class Handler:
    
    def check_timestamp(self, timestamp: int):
        should_be_enabled = (
            self.args.share_attention and 
            self.args.enable_attention_sharing and
            timestamp >= self.args.start_attention_sharing and 
            timestamp < self.args.stop_attention_sharing 
        )
        # print(timestamp, self.args.start_svd, self.args.end_svd)
        svd_should_be_enabled = (
            self.args.svd and 
            timestamp >= self.args.start_svd and 
            timestamp < self.args.end_svd 
        )
        # print(svd_should_be_enabled, self.args.svd_flag)
        # should_be_enabled = (
        #     self.args.share_attention and 
        #     (
        #         timestamp < self.args.start_attention_sharing or 
        #         timestamp >= self.args.stop_attention_sharing
        #     ) 
        # )
        
        processors = None
        if self.args.attention_sharing_steps_flag != should_be_enabled:
            self.args.attention_sharing_steps_flag = should_be_enabled
            processors = init_attention_processors(self.pipeline, self.args)
        
        if self.args.svd_flag != svd_should_be_enabled:
            # print(self.args.svd_flag)
            self.args.svd_flag = svd_should_be_enabled
            # print(self.args.svd_flag)
            processors = init_attention_processors(self.pipeline, self.args)
        
        return processors

    def register(self, style_aligned_args: StyleAlignedArgs, ):
        self.norm_layers = register_shared_norm(self.pipeline, style_aligned_args.share_group_norm,
                                                style_aligned_args.share_layer_norm)
        self.args = style_aligned_args
        self.check_timestamp(0)
        return init_attention_processors(self.pipeline, style_aligned_args)

    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None)

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.norm_layers = []
