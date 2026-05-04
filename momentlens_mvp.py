from __future__ import annotations

import os
import math
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer
from PIL import Image

try:
    from pytorchvideo.models.hub import slowfast_r50
except Exception as exc:  # pragma: no cover - import guard for local envs
    slowfast_r50 = None
    _PYTORCHVIDEO_IMPORT_ERROR = exc
else:
    _PYTORCHVIDEO_IMPORT_ERROR = None

CPU_THREADS = 4
try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    cv2.setNumThreads(CPU_THREADS)
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
IMPROVEMENT_CKPT_PATH = BASE_DIR / "Model" / "Improvement_Ckp" / "best.pt"
SF50_CKPT_PATH = BASE_DIR / "Model" / "SF50_Ckp" / "best.pt"

QUERY_BACKBONE_NAME = "distilbert-base-uncased"
FEATURE_DIM = 2304
FEATURE_CLIP_LEN_SEC = 1.0
NUM_FRAMES = 32
ALPHA = 4
MAX_QUERY_LEN = 32

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
        return ckpt_obj
    return ckpt_obj


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


class SharedFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2, num_heads: int = 8):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
            for _ in range(4)
        ])
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(4)
        ])
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = x
        mask_f = None if mask is None else mask.unsqueeze(-1).float()
        key_padding_mask = None if mask is None else ~mask.bool()

        if mask_f is not None:
            h = h * mask_f

        for conv, norm in zip(self.convs, self.conv_norms):
            y = conv(h.transpose(1, 2)).transpose(1, 2)
            h = norm(h + self.dropout(y))
            h = F.relu(h)
            if mask_f is not None:
                h = h * mask_f

        y, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.self_attn_norm(h + self.dropout(y))

        y = self.ffn(h)
        h = self.ffn_norm(h + self.dropout(y))

        if mask_f is not None:
            h = h * mask_f
        return h


class CrossModalGroundingModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        query_backbone_name: str = QUERY_BACKBONE_NAME,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.query_backbone_name = query_backbone_name
        self.query_backbone = AutoModel.from_pretrained(query_backbone_name)
        for p in self.query_backbone.parameters():
            p.requires_grad = False
        self.query_backbone.eval()

        query_hidden_dim = getattr(self.query_backbone.config, "hidden_size", None)
        if query_hidden_dim is None:
            query_hidden_dim = getattr(self.query_backbone.config, "dim", None)
        if query_hidden_dim is None:
            raise ValueError("Unable to infer query hidden size from pretrained model config")

        self.query_input_proj = nn.Linear(query_hidden_dim, hidden_dim)
        self.video_input_proj = nn.Linear(feature_dim, hidden_dim)
        self.shared_feature_encoder = SharedFeatureEncoder(hidden_dim, dropout=dropout)
        self.vsl_interaction_ffn = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.foreground_gate_conv = nn.Conv1d(hidden_dim * 2, 1, kernel_size=7, padding=3)

        self.start_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.end_lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.start_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.end_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.query_backbone.eval()
        return self

    def encode_query(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.query_backbone(input_ids=input_ids, attention_mask=attention_mask)
        q = self.query_input_proj(outputs.last_hidden_state)
        q = self.shared_feature_encoder(q, attention_mask.bool())
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (q * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return q, pooled

    def forward(self, features, video_mask, input_ids, attention_mask):
        bsz, t, _ = features.shape
        v = self.video_input_proj(features)
        v = self.shared_feature_encoder(v, video_mask)

        q_tokens, q_summary = self.encode_query(input_ids, attention_mask)
        q_mask = attention_mask.bool()

        similarity_scores = torch.matmul(v, q_tokens.transpose(1, 2)) / (v.size(-1) ** 0.5)
        sim_q = similarity_scores.masked_fill(~q_mask[:, None, :], float("-inf"))
        attn_q = F.softmax(sim_q, dim=-1)
        context_to_query = torch.matmul(attn_q, q_tokens)

        ctx_score = similarity_scores.max(dim=-1).values.masked_fill(~video_mask, float("-inf"))
        ctx_weight = F.softmax(ctx_score, dim=-1)
        global_video_context = torch.matmul(ctx_weight.unsqueeze(1), v).expand(-1, t, -1)

        video_query_features = self.vsl_interaction_ffn(
            torch.cat([v, context_to_query, v * context_to_query, v * global_video_context], dim=-1)
        )
        foreground_input = torch.cat([video_query_features, q_summary.unsqueeze(1).expand(-1, t, -1)], dim=-1)
        foreground_logits = self.foreground_gate_conv(foreground_input.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        foreground_gate = torch.sigmoid(foreground_logits)
        foreground_features = foreground_gate.unsqueeze(-1) * video_query_features

        lengths = video_mask.sum(dim=1).clamp_min(1).to(torch.long).cpu()
        packed_start = pack_padded_sequence(foreground_features, lengths, batch_first=True, enforce_sorted=False)
        packed_start_out, _ = self.start_lstm(packed_start)
        start_states, _ = pad_packed_sequence(packed_start_out, batch_first=True, total_length=t)
        start_logits = self.start_head(torch.cat([start_states, foreground_features], dim=-1)).squeeze(-1)

        end_input = torch.cat([foreground_features, start_states], dim=-1)
        packed_end = pack_padded_sequence(end_input, lengths, batch_first=True, enforce_sorted=False)
        packed_end_out, _ = self.end_lstm(packed_end)
        end_states, _ = pad_packed_sequence(packed_end_out, batch_first=True, total_length=t)
        end_logits = self.end_head(torch.cat([end_states, foreground_features], dim=-1)).squeeze(-1)

        neg_fill = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(~video_mask, neg_fill)
        end_logits = end_logits.masked_fill(~video_mask, neg_fill)
        foreground_logits = foreground_logits.masked_fill(~video_mask, neg_fill)
        return start_logits, end_logits, foreground_logits


@lru_cache(maxsize=1)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(QUERY_BACKBONE_NAME)


@lru_cache(maxsize=1)
def load_phase3_model():
    model = CrossModalGroundingModel(
        feature_dim=FEATURE_DIM,
        hidden_dim=128,
        query_backbone_name=QUERY_BACKBONE_NAME,
        dropout=0.2,
    )
    state = torch.load(IMPROVEMENT_CKPT_PATH, map_location="cpu")
    state_dict = _extract_state_dict(state)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    model.load_state_dict(state_dict, strict=True)
    return model


@lru_cache(maxsize=1)
def load_slowfast_extractor():
    if slowfast_r50 is None:
        raise RuntimeError(f"pytorchvideo is not available: {_PYTORCHVIDEO_IMPORT_ERROR}")

    if not SF50_CKPT_PATH.exists():
        raise FileNotFoundError(SF50_CKPT_PATH)

    model = slowfast_r50(pretrained=False)
    head = model.blocks[-1]
    if not hasattr(head, "proj"):
        raise RuntimeError("Unexpected SlowFast head layout")
    head.proj = nn.Linear(2304, 157)
    state = torch.load(SF50_CKPT_PATH, map_location="cpu")
    state_dict = _extract_state_dict(state)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    model.load_state_dict(state_dict, strict=True)

    head = model.blocks[-1]
    if not hasattr(head, "proj"):
        raise RuntimeError("Unexpected SlowFast head layout")
    head.proj = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def probe_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return fps, frame_count, duration


def write_predicted_clip(video_path: str, start_sec: float, end_sec: float, output_path: str):
    fps, _, duration = probe_video(video_path)
    fps = fps or 24.0
    start_sec = max(0.0, min(float(start_sec), float(duration)))
    end_sec = max(start_sec, min(float(end_sec), float(duration)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid video size for: {video_path}")
    out_width = max(2, width - (width % 2))
    out_height = max(2, height - (height % 2))

    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    current_frame = start_frame
    while current_frame < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        current_frame += 1

    cap.release()
    if not frames:
        raise RuntimeError("No frames available for predicted clip preview")
    duration_ms = max(20, int(round(1000.0 / max(fps, 1.0))))
    first, rest = frames[0], frames[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        format="GIF",
    )
    return output_path


def sample_windows(video_len_sec: float):
    full_clips = int(float(video_len_sec) // FEATURE_CLIP_LEN_SEC)
    if full_clips <= 0:
        return [(0.0, float(video_len_sec))]
    return [
        (float(i * FEATURE_CLIP_LEN_SEC), float((i + 1) * FEATURE_CLIP_LEN_SEC))
        for i in range(full_clips)
    ]


def read_window_frames(cap: cv2.VideoCapture, clip_start: float, clip_end: float, fps: float, num_frames: int = NUM_FRAMES):
    start_frame = max(0, int(math.floor(clip_start * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(clip_end * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    current_frame = start_frame
    while current_frame < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous().to(torch.uint8))
        current_frame += 1

    if not frames:
        raise RuntimeError(f"No frames decoded from window [{clip_start}, {clip_end}]")

    clip_frames = torch.stack(frames)
    if clip_frames.shape[0] < num_frames:
        pad = clip_frames[-1:].repeat(num_frames - clip_frames.shape[0], 1, 1, 1)
        clip_frames = torch.cat([clip_frames, pad], dim=0)
    elif clip_frames.shape[0] > num_frames:
        idx = np.linspace(0, clip_frames.shape[0] - 1, num_frames).astype(np.int64)
        clip_frames = clip_frames[idx]

    clip_frames = clip_frames.to(torch.float32).div(255.0)
    clip_frames = (clip_frames - KINETICS_MEAN) / KINETICS_STD
    return clip_frames


def _prepare_slowfast_inputs(clips: torch.Tensor):
    fast = clips.permute(0, 2, 1, 3, 4).contiguous()
    slow = fast[:, :, ::ALPHA, :, :].contiguous()
    return slow, fast


def extract_slowfast_features(video_path: str, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = load_slowfast_extractor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / fps if fps > 0 else 0.0
        windows = sample_windows(duration)

        clips = []
        starts = []
        ends = []
        for start_sec, end_sec in windows:
            clips.append(read_window_frames(cap, start_sec, end_sec, fps))
            starts.append(start_sec)
            ends.append(end_sec)
    finally:
        cap.release()

    feature_chunks = []
    with torch.inference_mode():
        for offset in range(0, len(clips), batch_size):
            clip_batch = torch.stack(clips[offset:offset + batch_size])
            slow, fast = _prepare_slowfast_inputs(clip_batch)
            features = extractor([slow.to(device, non_blocking=True), fast.to(device, non_blocking=True)])
            feature_chunks.append(features.detach().cpu().to(torch.float32))

    features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty(0, FEATURE_DIM)
    centers = torch.tensor([(s + e) / 2.0 for s, e in zip(starts, ends)], dtype=torch.float32)
    return {
        "features": features,
        "clip_starts": torch.tensor(starts, dtype=torch.float32),
        "clip_ends": torch.tensor(ends, dtype=torch.float32),
        "centers": centers,
        "duration": float(duration),
        "fps": float(fps),
    }


def masked_soft_cross_entropy(logits, target, mask):
    target = target * mask.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(logits, dim=-1)
    return (-(target * log_probs).sum(dim=-1)).mean()


def select_span(start_logits, end_logits, video_mask):
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    score = start_probs[:, :, None] * end_probs[:, None, :]

    bsz, t, _ = score.shape
    idx = torch.arange(t, device=score.device)
    tri_mask = idx[None, :, None] <= idx[None, None, :]
    valid = tri_mask.expand(bsz, -1, -1) & video_mask[:, :, None] & video_mask[:, None, :]

    score = score.masked_fill(~valid, float("-inf"))
    flat = score.view(bsz, -1)
    best = flat.argmax(dim=-1)
    has_valid = valid.view(bsz, -1).any(dim=-1)

    fallback_start = start_probs.masked_fill(~video_mask, float("-inf")).argmax(dim=-1)
    fallback_end_scores = end_probs.masked_fill(~video_mask, float("-inf"))
    fallback_end_scores = fallback_end_scores.masked_fill(idx[None, :] < fallback_start[:, None], float("-inf"))
    fallback_end = fallback_end_scores.argmax(dim=-1)

    start = best // t
    end = best % t
    if not has_valid.all():
        start = torch.where(has_valid, start, fallback_start)
        end = torch.where(has_valid, end, fallback_end)
    return start, end


@lru_cache(maxsize=1)
def load_phase3_runtime():
    model = load_phase3_model()
    tokenizer = load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, device


def predict_moment(video_path: str, query: str):
    model, tokenizer, device = load_phase3_runtime()
    extracted = extract_slowfast_features(video_path)

    features = extracted["features"].unsqueeze(0).to(device)
    video_mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=device)
    centers = extracted["centers"].to(device).unsqueeze(0)

    tokens = tokenizer(
        query,
        max_length=MAX_QUERY_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.inference_mode():
        start_logits, end_logits, foreground_logits = model(features, video_mask, input_ids, attention_mask)
        start_idx, end_idx = select_span(start_logits, end_logits, video_mask)

    start_sec = float(centers.gather(1, start_idx.unsqueeze(1)).squeeze(1).item())
    end_sec = float(centers.gather(1, end_idx.unsqueeze(1)).squeeze(1).item())
    return {
        "duration": extracted["duration"],
        "fps": extracted["fps"],
        "num_windows": int(features.shape[1]),
        "start_idx": int(start_idx.item()),
        "end_idx": int(end_idx.item()),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "timeline_centers": extracted["centers"].tolist(),
        "start_logits": start_logits.squeeze(0).detach().cpu().tolist(),
        "end_logits": end_logits.squeeze(0).detach().cpu().tolist(),
        "foreground_logits": foreground_logits.squeeze(0).detach().cpu().tolist(),
        "features_shape": list(features.shape),
    }


def render_timeline(duration: float, start_sec: float, end_sec: float):
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.barh([0], [duration], left=[0], height=0.34, color="#e5e7eb", edgecolor="#d1d5db")
    ax.barh([0], [max(0.0, end_sec - start_sec)], left=[start_sec], height=0.34, color="#22c55e", alpha=0.9)
    ax.axvline(start_sec, color="#15803d", linewidth=2)
    ax.axvline(end_sec, color="#15803d", linewidth=2)
    ax.set_xlim(0, max(duration, end_sec + 0.5))
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Predicted moment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig
