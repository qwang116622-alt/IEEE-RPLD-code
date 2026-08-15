"""GCN-Former-TKAN model components used in the manuscript configuration."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """One normalized graph-convolution operation, ``A_hat X W + b``."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adjacency, support)
        return output if self.bias is None else output + self.bias


class KANLayer(nn.Module):
    """KAN layer with learnable cubic B-spline connection functions.

    ``grid_size`` is the number of intervals and ``spline_order`` is the
    B-spline degree. The manuscript configuration uses a grid of 5 and a
    third-order (cubic) B-spline.
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        if grid_size < 1 or spline_order < 1:
            raise ValueError("grid_size and spline_order must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order

        interior = torch.linspace(-1.0, 1.0, grid_size + 1)
        knots = torch.cat(
            [interior[:1].repeat(spline_order), interior, interior[-1:].repeat(spline_order)]
        )
        self.register_buffer("knots", knots)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_basis)
        )
        self.spline_scale = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def _bspline_basis(self, x):
        """Evaluate B-spline bases for ``x`` with shape ``(B, T, I)``."""
        x = torch.tanh(x).clamp(min=-0.999999, max=0.999999)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        bases = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).to(x.dtype)

        for degree in range(1, self.spline_order + 1):
            n_basis = knots.numel() - degree - 1
            left_denominator = knots[degree:-1] - knots[:-degree - 1]
            right_denominator = knots[degree + 1:] - knots[1:-degree]
            left_numerator = x.unsqueeze(-1) - knots[:-degree - 1]
            right_numerator = knots[degree + 1:] - x.unsqueeze(-1)
            # Repeated boundary knots deliberately create zero denominators.
            # Their Cox-de Boor contribution is zero, so avoid 0/0 NaNs.
            left_valid = left_denominator.abs() > 1e-12
            right_valid = right_denominator.abs() > 1e-12
            safe_left_denominator = torch.where(
                left_valid, left_denominator, torch.ones_like(left_denominator)
            )
            safe_right_denominator = torch.where(
                right_valid, right_denominator, torch.ones_like(right_denominator)
            )
            left = (left_numerator / safe_left_denominator) * left_valid.to(x.dtype)
            right = (right_numerator / safe_right_denominator) * right_valid.to(x.dtype)
            bases = left * bases[..., :n_basis] + right * bases[..., 1:n_basis + 1]
        return bases

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected (batch, sequence, {self.in_features}), got {tuple(x.shape)}"
            )
        base_output = F.linear(F.silu(x), self.base_weight, self.bias)
        bases = self._bspline_basis(x)
        scaled_spline_weight = self.spline_weight * self.spline_scale.unsqueeze(-1)
        spline_output = torch.einsum("btik,oik->bto", bases, scaled_spline_weight)
        return base_output + spline_output


class TemporalKANLayer(nn.Module):
    """One gated recurrent KAN layer with three KAN units.

    The KAN units form the nonlinear recurrent candidate mapping. Input,
    forget, and output gates retain the temporal memory behaviour described for
    TKAN in the manuscript.
    """

    def __init__(self, input_dim, hidden_dim, num_kan_units, grid_size, spline_order, dropout):
        super().__init__()
        if num_kan_units != 3:
            raise ValueError("The manuscript architecture requires three KAN units per TKAN layer")

        dimensions = [input_dim + hidden_dim] + [hidden_dim] * num_kan_units
        self.kan_units = nn.ModuleList(
            KANLayer(dimensions[i], dimensions[i + 1], grid_size, spline_order)
            for i in range(num_kan_units)
        )
        gate_input_dim = input_dim + hidden_dim
        self.input_gate = nn.Linear(gate_input_dim, hidden_dim)
        self.forget_gate = nn.Linear(gate_input_dim, hidden_dim)
        self.output_gate = nn.Linear(gate_input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        hidden = x.new_zeros(batch_size, self.hidden_dim)
        cell = x.new_zeros(batch_size, self.hidden_dim)
        outputs = []

        for step in range(sequence_length):
            gate_input = torch.cat([x[:, step], hidden], dim=-1)
            candidate = gate_input.unsqueeze(1)
            for kan_unit in self.kan_units:
                candidate = F.silu(kan_unit(candidate))
            candidate = candidate.squeeze(1)

            input_gate = torch.sigmoid(self.input_gate(gate_input))
            forget_gate = torch.sigmoid(self.forget_gate(gate_input))
            output_gate = torch.sigmoid(self.output_gate(gate_input))
            cell = forget_gate * cell + input_gate * torch.tanh(candidate)
            hidden = output_gate * torch.tanh(cell)
            outputs.append(self.dropout(hidden))

        return torch.stack(outputs, dim=1)


class TKAN(nn.Module):
    """Two-layer temporal KAN encoder used for nonlinear sequence mapping."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        num_kan_units=3,
        grid_size=5,
        spline_order=3,
        dropout=0.0,
    ):
        super().__init__()
        if num_layers != 2:
            raise ValueError("The manuscript architecture requires two TKAN layers")
        if output_dim != hidden_dim:
            raise ValueError("TKAN output_dim must equal hidden_dim before the final prediction head")

        self.layers = nn.ModuleList(
            [
                TemporalKANLayer(
                    input_dim if index == 0 else hidden_dim,
                    hidden_dim,
                    num_kan_units,
                    grid_size,
                    spline_order,
                    dropout,
                )
                for index in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.output_norm(x)
        return output, output[:, -1, :]


class PositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional encoding for temporal ordering."""

    def __init__(self, embedding_dim, max_length=512):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError("Sequence length exceeds positional-encoding capacity")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class TransformerEncoder(nn.Module):
    """Layer-normalized, position-aware Transformer temporal encoder."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        dropout=0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers != 2:
            raise ValueError("The manuscript architecture requires two Transformer encoder layers")

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_projection(x)
        return self.encoder(self.position_encoding(x))


class GCN(nn.Module):
    """Twenty-node Spearman-topology GCN with two 32-dimensional convolutions."""

    def __init__(self, gcn_config, adjacency_matrix=None):
        super().__init__()
        self.num_nodes = gcn_config["num_nodes"]
        self.output_dim = gcn_config["output_dim"]
        num_layers = gcn_config["num_layers"]
        if num_layers != 2:
            raise ValueError("The manuscript architecture requires two GCN layers")

        adjacency = self._normalize_adjacency(adjacency_matrix)
        self.register_buffer("adjacency", adjacency)
        dimensions = [gcn_config["input_dim"], gcn_config["hidden_dim"], gcn_config["output_dim"]]
        self.layers = nn.ModuleList(
            GraphConvolution(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.dropout = nn.Dropout(gcn_config["dropout"])

    def _normalize_adjacency(self, adjacency_matrix):
        if adjacency_matrix is None:
            adjacency = torch.eye(self.num_nodes, dtype=torch.float32)
        else:
            adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32)
            if adjacency.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Expected a {self.num_nodes}x{self.num_nodes} adjacency matrix, got {tuple(adjacency.shape)}"
                )
        adjacency = adjacency + torch.eye(self.num_nodes, dtype=adjacency.dtype)
        degrees = adjacency.sum(dim=1).clamp_min(1e-12)
        inverse_sqrt_degree = torch.diag(degrees.pow(-0.5))
        return inverse_sqrt_degree @ adjacency @ inverse_sqrt_degree

    def forward(self, x):
        if x.ndim != 2 or x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected (batch, {self.num_nodes}) pile features, got {tuple(x.shape)}"
            )
        x = x.unsqueeze(-1)
        for index, layer in enumerate(self.layers):
            x = layer(x, self.adjacency)
            if index < len(self.layers) - 1:
                x = self.dropout(F.relu(x))
        return x.mean(dim=1)


class PileDisplacementModel(nn.Module):
    """Hierarchical GCN -> Transformer -> TKAN retaining-pile predictor."""

    def __init__(self, gcn_config, tkan_config, transformer_config, adj_matrix=None):
        super().__init__()
        self.gcn = GCN(gcn_config, adj_matrix)
        self.transformer = TransformerEncoder(
            input_dim=gcn_config["output_dim"],
            hidden_dim=transformer_config["hidden_dim"],
            num_heads=transformer_config["num_heads"],
            num_layers=transformer_config["num_layers"],
            feedforward_dim=transformer_config["feedforward_dim"],
            dropout=transformer_config["dropout"],
        )

        raw_feature_dim = (
            tkan_config["jg_input_dim"]
            + tkan_config["aq_input_dim"]
            + tkan_config["j_input_dim"]
        )
        self.tkan = TKAN(
            input_dim=raw_feature_dim,
            hidden_dim=tkan_config["hidden_dim"],
            num_layers=tkan_config["num_layers"],
            output_dim=tkan_config["output_dim"],
            num_kan_units=tkan_config["num_kan_units"],
            grid_size=tkan_config["grid_size"],
            spline_order=tkan_config["spline_order"],
            dropout=tkan_config["dropout"],
        )
        self.spatial_projection = nn.Linear(
            transformer_config["hidden_dim"], tkan_config["hidden_dim"]
        )
        self.fusion_norm = nn.LayerNorm(tkan_config["hidden_dim"])
        self.prediction_head = nn.Linear(tkan_config["output_dim"], 1)

    def forward(self, features):
        graph_pile_features = features["j1_j20"]
        non_target_pile_features = features["j_non_target"]
        _, sequence_length, _ = graph_pile_features.shape
        spatial_sequence = torch.stack(
            [self.gcn(graph_pile_features[:, step, :]) for step in range(sequence_length)],
            dim=1,
        )
        enhanced_spatial = self.transformer(spatial_sequence)
        raw_tkan_features = torch.cat(
            [
                features["jg1_jg11"],
                features["aq1_aq17"],
                non_target_pile_features,
            ],
            dim=-1,
        )
        tkan_sequence, _ = self.tkan(raw_tkan_features)
        spatial_context = self.spatial_projection(enhanced_spatial)
        fused_sequence = self.fusion_norm(tkan_sequence + spatial_context)
        return self.prediction_head(fused_sequence[:, -1, :])

