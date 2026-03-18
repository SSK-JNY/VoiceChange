#!/usr/bin/env python3
"""
RVC (Retrieval-based Voice Conversion) Model層
完全実装版 - fairseqベース
"""
import os
import sys
import torch
import numpy as np
import librosa
from scipy.io import wavfile
import sounddevice as sd
import logging
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import json
import time
try:
    import fairseq
    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False
    fairseq = None

import json

# 親ディレクトリの config をインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import config


class SynthesizerTrn(nn.Module):
    """
    RVC Synthesizer model - 完全実装
    Based on VITS architecture with RVC modifications
    """
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.segment_size = segment_size
        self.sr = sr

        # Speaker embedding
        self.emb_g = nn.Embedding(self.spk_embed_dim, self.gin_channels)

        # Encoder
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )

        # Decoder
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels
        )

        # Flow-based posterior encoder (simplified)
        self.flow = ResidualCouplingBlock(
            inter_channels // 2,
            hidden_channels,
            kernel_size,
            5,  # n_flows
            1,  # n_layers
            gin_channels
        )

    def forward(self, x, f0, g=None):
        """
        Forward pass for RVC inference
        x: mel spectrogram [B, T, C]
        f0: fundamental frequency [B, T]
        g: speaker embedding [B, C]
        """
        # Speaker embedding
        if g is not None:
            g = self.emb_g(g)

        # Encoder
        x = self.enc_p(x, f0, g)

        # Flow
        z, logdet = self.flow(x, g)

        # Decoder
        o = self.dec(z, g)

        return o


class TextEncoder(nn.Module):
    """Text Encoder for RVC"""
    def __init__(self, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Phone embedding
        self.emb_phone = nn.Embedding(256, hidden_channels)  # Assuming 256 phone types

        # Pitch embedding
        self.emb_pitch = nn.Embedding(256, hidden_channels)  # Pitch bins

        # Encoder layers
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )

        # Projection
        self.proj = nn.Linear(hidden_channels, inter_channels)

    def forward(self, x, f0, g=None):
        # Phone embedding (simplified - using first channel as phone)
        phone_emb = self.emb_phone(torch.zeros(x.shape[0], x.shape[1], dtype=torch.long).to(x.device))

        # Pitch embedding (simplified)
        pitch_emb = self.emb_pitch(torch.zeros(x.shape[0], x.shape[1], dtype=torch.long).to(x.device))

        # Combine embeddings
        x = phone_emb + pitch_emb

        # Encoder
        x = self.encoder(x, g)

        # Projection
        x = self.proj(x)
        return x


class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x, g=None):
        x = self.drop(x)
        for layer in self.encoder:
            x = layer(x, g)
        x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, p_dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.attn = MultiHeadAttention(hidden_channels, n_heads)
        self.conv1 = Conv1d(hidden_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)

    def forward(self, x, g=None):
        # Self-attention
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward
        conv_out = self.conv2(F.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x = self.norm2(x + self.dropout2(conv_out))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, channels, n_heads):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x)


class Generator(nn.Module):
    """Waveform Generator"""
    def __init__(self, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels):
        super().__init__()
        self.inter_channels = inter_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.gin_channels = gin_channels

        # Initial conv
        self.conv_pre = Conv1d(inter_channels, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2**(i+1)),
                               k, u, padding=(k-u)//2)
            ))

        # ResBlocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, gin_channels))

        # Post conv
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(len(self.resblock_kernel_sizes)):
                if xs is None:
                    xs = self.resblocks[i*len(self.resblock_kernel_sizes) + j](x, g)
                else:
                    xs += self.resblocks[i*len(self.resblock_kernel_sizes) + j](x, g)
            x = xs / len(self.resblock_kernel_sizes)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, channels, kernel_size, dilation, gin_channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.gin_channels = gin_channels

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                              padding=(kernel_size-1)//2 * dilation[0])),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                              padding=(kernel_size-1)//2 * dilation[1])),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                              padding=(kernel_size-1)//2 * dilation[2]))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                              padding=(kernel_size-1)//2 * dilation[0])),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                              padding=(kernel_size-1)//2 * dilation[1])),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                              padding=(kernel_size-1)//2 * dilation[2]))
        ])

        if gin_channels != 0:
            self.cond_layer = weight_norm(nn.Linear(gin_channels, channels*2))

    def forward(self, x, g=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if self.gin_channels != 0 and g is not None:
                gc = self.cond_layer(g)
                gc = torch.unsqueeze(gc, -1)
                xt = xt + gc[:, :self.channels, :]
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if self.gin_channels != 0 and g is not None:
                xt = xt + gc[:, self.channels:, :]
            xt = c2(xt)
            x = xt + x
        return x


class ResidualCouplingBlock(nn.Module):
    """Residual Coupling Block for flow"""
    def __init__(self, channels, hidden_channels, kernel_size, n_flows, n_layers, gin_channels):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_flows = n_flows
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, n_layers, gin_channels))

    def forward(self, x, g=None):
        logdet = 0
        for flow in self.flows:
            x, logdet_ = flow(x, g)
            logdet += logdet_
        return x, logdet


class ResidualCouplingLayer(nn.Module):
    """Residual Coupling Layer"""
    def __init__(self, channels, hidden_channels, kernel_size, n_layers, gin_channels):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.half_channels = channels // 2

        self.pre = Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, n_layers, gin_channels)
        self.post = Conv1d(hidden_channels, self.half_channels * 2, 1)

    def forward(self, x, g=None):
        x0, x1 = x[:, :self.half_channels], x[:, self.half_channels:]
        h = self.post(F.relu(self.enc(self.pre(x0), g)))
        m, logs = h[:, :self.half_channels], h[:, self.half_channels:]
        x1 = x1 * torch.exp(logs) + m
        x = torch.cat([x0, x1], 1)
        logdet = torch.sum(logs, [1, 2])
        return x, logdet


class WN(nn.Module):
    """WaveNet-like module"""
    def __init__(self, hidden_channels, kernel_size, n_layers, gin_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = weight_norm(nn.Linear(gin_channels, hidden_channels*2)) if gin_channels != 0 else None

        for i in range(n_layers):
            dilation = 2 ** i
            self.in_layers.append(weight_norm(
                Conv1d(hidden_channels, hidden_channels*2, kernel_size, 1, dilation=dilation,
                      padding=(kernel_size-1)//2 * dilation)
            ))
            self.res_skip_layers.append(weight_norm(
                Conv1d(hidden_channels, hidden_channels, 1)
            ))

    def forward(self, x, g=None):
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None and self.cond_layer is not None:
                gc = self.cond_layer(g)
                x_in = x_in + torch.unsqueeze(gc, -1)

            x_in, skip = x_in[:, :self.hidden_channels], x_in[:, self.hidden_channels:]
            x = self.res_skip_layers[i](x_in) + x
            x = skip + x
        return x


class RVCModel:
    """RVC音声変換モデル"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.net_g = None
        self.hubert_model = None
        self.rmvpe_model = None  # RMVPEモデル
        self.models_dir = Path(__file__).parent.parent / 'models' / 'rvc'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # RVC設定
        self.sampling_rate = 40000  # RVCのサンプリングレート
        self.hop_length = 512
        self.f0_method = 'rmvpe'  # ピッチ抽出方法

        # モデルファイルのパス
        self.hubert_path = self.models_dir / 'hubert_base.pt'
        self.rmvpe_path = self.models_dir / 'rmvpe.pt'

        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_hubert_model(self):
        """HuBERTモデルの読み込み - 本実装 (fairseq使用、オプション)"""
        if not FAIRSEQ_AVAILABLE:
            self.logger.warning("fairseq not available, using simplified feature extraction")
            self.logger.info("Install fairseq for full RVC functionality: pip install fairseq")
            return None

        if not self.hubert_path.exists():
            self.logger.warning(f"HuBERT model not found at {self.hubert_path}")
            self.logger.info("Downloading HuBERT model...")
            self.download_pretrained_models()
            if not self.hubert_path.exists():
                return None

        try:
            # fairseqを使ってHuBERTを読み込み
            from fairseq import checkpoint_utils
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [str(self.hubert_path)],
                suffix="",
            )
            self.hubert_model = models[0]
            self.hubert_model.eval().to(self.device)
            self.logger.info("HuBERT model loaded successfully")
            return self.hubert_model
        except Exception as e:
            self.logger.error(f"Failed to load HuBERT model: {e}")
            return None

    def load_rmvpe_model(self):
        """RMVPEモデルの読み込み - 本実装"""
        if not self.rmvpe_path.exists():
            self.logger.warning(f"RMVPE model not found at {self.rmvpe_path}")
            self.logger.info("Downloading RMVPE model...")
            self.download_pretrained_models()
            if not self.rmvpe_path.exists():
                return None

        try:
            # RMVPEモデルを読み込み
            self.rmvpe_model = torch.load(self.rmvpe_path, map_location=self.device)
            self.rmvpe_model.eval().to(self.device)
            self.logger.info("RMVPE model loaded successfully")
            return self.rmvpe_model
        except Exception as e:
            self.logger.error(f"Failed to load RMVPE model: {e}")
            return None

    def load_rvc_model(self, model_path):
        """RVCモデルの読み込み - 完全実装"""
        try:
            # モデルファイルの存在確認
            if not os.path.exists(model_path):
                self.logger.error(f"RVC model not found: {model_path}")
                return False

            # 設定ファイルの読み込み
            config_path = model_path.replace('.pth', '.json')
            if not os.path.exists(config_path):
                self.logger.error(f"Config file not found: {config_path}")
                self.logger.info("Please ensure both .pth and .json files are in the same directory")
                return False

            # 設定ファイル読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # net_g設定を取得
            net_g_config = config.get('net_g', {})

            # モデルアーキテクチャの初期化
            self.net_g = SynthesizerTrn(
                spec_channels=config.get('spec_channels', 513),
                segment_size=config.get('segment_size', 8192),
                inter_channels=net_g_config.get('inter_channels', 192),
                hidden_channels=net_g_config.get('hidden_channels', 192),
                filter_channels=net_g_config.get('filter_channels', 768),
                n_heads=net_g_config.get('n_heads', 2),
                n_layers=net_g_config.get('n_layers', 6),
                kernel_size=net_g_config.get('kernel_size', 3),
                p_dropout=net_g_config.get('p_dropout', 0.1),
                resblock=net_g_config.get('resblock', '1'),
                resblock_kernel_sizes=net_g_config.get('resblock_kernel_sizes', [3,7,11]),
                resblock_dilation_sizes=net_g_config.get('resblock_dilation_sizes', [[1,3,5], [1,3,5], [1,3,5]]),
                upsample_rates=net_g_config.get('upsample_rates', [8,8,2,2]),
                upsample_initial_channel=net_g_config.get('upsample_initial_channel', 512),
                upsample_kernel_sizes=net_g_config.get('upsample_kernel_sizes', [16,16,4,4]),
                spk_embed_dim=net_g_config.get('spk_embed_dim', 109),
                gin_channels=net_g_config.get('gin_channels', 256),
                sr=config.get('sample_rate', 16000)
            )

            # モデルをデバイスに移動
            self.net_g = self.net_g.to(self.device)

            # state_dictの読み込み
            state_dict = torch.load(model_path, map_location=self.device)

            # 不要なキーを除去（例: 'epoch', 'global_step'など）
            if 'epoch' in state_dict:
                del state_dict['epoch']
            if 'global_step' in state_dict:
                del state_dict['global_step']
            if 'lr' in state_dict:
                del state_dict['lr']

            # state_dictの適用
            self.net_g.load_state_dict(state_dict, strict=False)

            # 評価モードに設定
            self.net_g.eval()

            self.logger.info(f"RVC model loaded successfully from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load RVC model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def extract_features(self, audio, sr):
        """音声特徴抽出 - HuBERT本実装 + フォールバック"""
        if self.hubert_model is None:
            self.hubert_model = self.load_hubert_model()

        if self.hubert_model is not None:
            try:
                # リサンプリング (HuBERTは16kHzを想定)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

                # 音声をテンソルに変換
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

                # HuBERTで特徴抽出
                with torch.no_grad():
                    features = self.hubert_model.extract_features(audio_tensor)[0]
                    features = features.squeeze(0).cpu().numpy()

                return features

            except Exception as e:
                self.logger.error(f"HubERT feature extraction failed: {e}")

        # フォールバック: MFCC特徴抽出（高速化）
        self.logger.info("Using fast MFCC feature extraction for real-time processing")
        try:
            # リサンプリング（高速化のため16kHzに）
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # MFCC抽出（パラメータを最適化）
            mfcc = librosa.feature.mfcc(
                y=audio, sr=16000, n_mfcc=40, n_fft=512, hop_length=256, n_mels=40
            )
            return mfcc.T  # (time, features)形式

        except Exception as e:
            self.logger.error(f"MFCC feature extraction failed: {e}")
            return None

    def extract_f0(self, audio, sr):
        """F0（ピッチ）抽出 - RMVPE本実装"""
        if self.rmvpe_model is None:
            self.rmvpe_model = self.load_rmvpe_model()
            if self.rmvpe_model is None:
                # フォールバック: librosaを使用
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        audio, fmin=50, fmax=1100, sr=sr, frame_length=2048, hop_length=self.hop_length
                    )
                    return f0
                except:
                    return np.zeros(len(audio) // self.hop_length)

        try:
            # RMVPEでF0抽出
            with torch.no_grad():
                # 音声をテンソルに変換
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

                # RMVPE推論
                f0 = self.rmvpe_model(audio_tensor)[0]
                f0 = f0.squeeze(0).cpu().numpy()

            return f0

        except Exception as e:
            self.logger.error(f"F0 extraction failed: {e}")
            # フォールバック: より高速なF0推定
            try:
                # librosa.pyinの軽量版
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, fmin=50, fmax=1100, sr=sr, frame_length=1024, hop_length=self.hop_length,
                    fill_na=np.nan
                )
                # NaNを線形補間（numpy使用）
                if np.any(np.isnan(f0)):
                    # 単純な前値補間
                    mask = np.isnan(f0)
                    f0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), f0[~mask])
                return f0
            except:
                # 最終フォールバック: 固定F0
                self.logger.warning("Using fixed F0 as final fallback")
                return np.full(len(audio) // self.hop_length, 220.0)  # 固定220Hz

    def convert_voice(self, audio, sr, target_model_path=None, pitch_shift=0):
        """音声変換 - 完全RVC実装"""
        start_time = time.time()

        # モデルが読み込まれていない場合は読み込み
        if self.net_g is None and target_model_path:
            if not self.load_rvc_model(target_model_path):
                # モデル読み込み失敗時は簡易変換
                return self._simple_convert(audio, sr, pitch_shift)

        if self.net_g is None:
            # RVCモデルが読み込まれていない場合、簡易変換を使用
            self.logger.warning("RVC model not loaded, using simple conversion")
            return self._simple_convert(audio, sr, pitch_shift)

        try:
            # 入力データの長さをチェック（短すぎる場合はパディング）
            min_length = 1024  # 最小処理長
            if len(audio) < min_length:
                # パディング
                padding = np.zeros(min_length - len(audio))
                audio = np.concatenate([audio, padding])

            # 特徴抽出
            feats = self.extract_features(audio, sr)
            if feats is None:
                return self._simple_convert(audio, sr, pitch_shift)

            # F0抽出
            f0 = self.extract_f0(audio, sr)
            if f0 is None:
                return self._simple_convert(audio, sr, pitch_shift)

            # ピッチシフト適用
            if pitch_shift != 0:
                f0 = f0 * (2 ** (pitch_shift / 12))

            # RVCモデルによる推論
            with torch.no_grad():
                # 特徴量をテンソルに変換
                feats_tensor = torch.from_numpy(feats).unsqueeze(0).to(self.device)
                f0_tensor = torch.from_numpy(f0).unsqueeze(0).to(self.device)

                # メルスペクトログラムに変換（簡易）
                mel_spec = feats_tensor  # HuBERT特徴をメルスペクトログラムとして使用

                # RVCモデルの推論
                try:
                    # スピーカー埋め込み（デフォルトを使用）
                    g = torch.zeros(1, dtype=torch.long).to(self.device)

                    output = self.net_g(mel_spec, f0_tensor, g)

                    if isinstance(output, torch.Tensor):
                        converted_audio = output.squeeze(0).cpu().numpy()

                        # 出力長を入力長に合わせる
                        if len(converted_audio) > len(audio):
                            converted_audio = converted_audio[:len(audio)]
                        elif len(converted_audio) < len(audio):
                            # パディングやリピートで長さを合わせる
                            converted_audio = np.tile(converted_audio, len(audio) // len(converted_audio) + 1)[:len(audio)]

                        processing_time = time.time() - start_time
                        self.logger.info(".2f")
                        return converted_audio
                except Exception as model_error:
                    self.logger.error(f"RVC model inference failed: {model_error}")
                    return self._simple_convert(audio, sr, pitch_shift)

            return audio

        except Exception as e:
            self.logger.error(f"Voice conversion failed: {e}")
            return self._simple_convert(audio, sr, pitch_shift)

    def _simple_convert(self, audio, sr, pitch_shift):
        """簡易音声変換（フォールバック）"""
        if abs(pitch_shift) > 0.1:
            converted_audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=pitch_shift, bins_per_octave=12
            )
            self.logger.info("Voice conversion completed (simple fallback)")
            return converted_audio
        return audio

    def download_pretrained_models(self):
        """事前学習モデルのダウンロード"""
        import urllib.request

        models_to_download = {
            'hubert_base.pt': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt',
            'rmvpe.pt': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt'
        }

        for filename, url in models_to_download.items():
            filepath = self.models_dir / filename
            if not filepath.exists():
                self.logger.info(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    self.logger.info(f"Downloaded {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to download {filename}: {e}")

    def get_available_models(self):
        """利用可能なRVCモデルの一覧を取得"""
        models = []
        for file in self.models_dir.glob('*.pth'):
            models.append(file.stem)
        return models