"""Optional torch models for contender backends."""

from __future__ import annotations


def build_cnn(contender_name: str, *, channels: int, num_classes: int):
    """Build small CNN baseline."""
    import torch.nn as nn

    dropout = 0.0
    hidden = 64
    if contender_name == "cnn_medium":
        hidden = 96
    elif contender_name == "cnn_regularized":
        hidden = 96
        dropout = 0.25

    layers: list[nn.Module] = [
        nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(32 * 2 * 2, hidden),
        nn.ReLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, num_classes))
    return nn.Sequential(*layers)


def build_transformer_lm(contender_name: str, *, vocab_size: int, context_length: int):
    """Build tiny decoder-only LM."""
    import torch
    import torch.nn as nn

    if contender_name == "transformer_lm_small":
        d_model = 96
        nhead = 4
        layers = 3
        ff = 192
        dropout = 0.1
    else:
        d_model = 64
        nhead = 4
        layers = 2
        ff = 128
        dropout = 0.1

    class TinyTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(context_length, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = self.token_embed(tokens) + self.pos_embed(positions)[None, :, :]
            mask = torch.triu(
                torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device),
                diagonal=1,
            )
            hidden = self.encoder(hidden, mask=mask)
            hidden = self.norm(hidden)
            return self.head(hidden)

    return TinyTransformerLM()
