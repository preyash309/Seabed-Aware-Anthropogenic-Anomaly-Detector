from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

FLOW_ROOT = Path(
    r"E:\SIH\SIH_Results\latent_normalizing_flow"
)

FLOW_CHECKPOINT = (
    FLOW_ROOT / "best_flow.pt"
)

LATENT_MEAN_PATH = (
    FLOW_ROOT / "latent_mean.pt"
)

LATENT_STD_PATH = (
    FLOW_ROOT / "latent_std.pt"
)


DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


LATENT_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 8
SCALE_CLAMP = 1.5

# ============================================================
# AFFINE COUPLING LAYER
# ============================================================

class AffineCoupling(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_clamp: float = 1.5,
    ):
        super().__init__()

        self.dim = dim
        self.scale_clamp = scale_clamp

        self.register_buffer(
            "mask",
            mask,
        )

        active_dim = int(
            mask.sum().item()
        )

        transformed_dim = (
            dim - active_dim
        )

        self.net = nn.Sequential(

            nn.Linear(
                active_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                2 * transformed_dim,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        mask_bool = (
            self.mask > 0.5
        )

        x_active = x[
            :,
            mask_bool,
        ]

        params = self.net(
            x_active
        )

        transformed_dim = (
            self.dim
            -
            int(
                self.mask.sum()
                .item()
            )
        )

        s, t = torch.chunk(
            params,
            2,
            dim=1,
        )

        # ----------------------------------------------------
        # Bounded scale.
        # ----------------------------------------------------

        s = torch.tanh(s)

        s = (
            s
            *
            self.scale_clamp
        )

        output = x.clone()

        transform_mask = (
            ~mask_bool
        )

        x_transformed = x[
            :,
            transform_mask,
        ]

        output[
            :,
            transform_mask
        ] = (
            x_transformed
            *
            torch.exp(s)
            +
            t
        )

        log_det = s.sum(
            dim=1
        )

        return output, log_det


# ============================================================
# REALNVP
# ============================================================

class RealNVP(nn.Module):

    def __init__(
        self,
        dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        scale_clamp: float = SCALE_CLAMP,
    ):
        super().__init__()

        layers = []

        # ----------------------------------------------------
        # Alternating binary masks.
        #
        # Layer 0:
        # 1 0 1 0 ...
        #
        # Layer 1:
        # 0 1 0 1 ...
        # ----------------------------------------------------

        for layer_index in range(
            num_layers
        ):

            mask = torch.zeros(
                dim,
                dtype=torch.float32,
            )

            if layer_index % 2 == 0:

                mask[::2] = 1.0

            else:

                mask[1::2] = 1.0

            layers.append(
                AffineCoupling(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    scale_clamp=scale_clamp,
                )
            )

        self.layers = nn.ModuleList(
            layers
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        z = x

        total_log_det = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        for layer in self.layers:

            z, log_det = layer(
                z
            )

            total_log_det += (
                log_det
            )

        return (
            z,
            total_log_det,
        )


# ============================================================
# STANDARD NORMAL LOG PROBABILITY
# ============================================================

def standard_normal_log_prob(
    z: torch.Tensor,
) -> torch.Tensor:

    return (
        -0.5
        *
        (
            z.pow(2)
            +
            torch.log(
                torch.tensor(
                    2.0
                    *
                    torch.pi,
                    device=z.device,
                    dtype=z.dtype,
                )
            )
        )
    ).sum(
        dim=1
    )


# ============================================================
# NLL
# ============================================================

@torch.inference_mode()
def flow_nll(
    model: RealNVP,
    x: torch.Tensor,
) -> torch.Tensor:

    z, log_det = model(
        x
    )

    log_prob = (
        standard_normal_log_prob(
            z
        )
        +
        log_det
    )

    return -log_prob


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def _extract_state_dict(
    checkpoint,
):

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in [
            "model_state_dict",
            "state_dict",
            "flow_state_dict",
            "model",
        ]:

            value = checkpoint.get(
                key
            )

            if isinstance(
                value,
                dict,
            ):

                return value

        return checkpoint

    raise RuntimeError(
        "Could not locate RealNVP state_dict."
    )


# ============================================================
# LOAD REALNVP
# ============================================================

def load_realnvp():

    for path in [
        FLOW_CHECKPOINT,
        LATENT_MEAN_PATH,
        LATENT_STD_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required RealNVP artifact "
                f"not found:\n{path}"
            )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RealNVP(
        dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        scale_clamp=SCALE_CLAMP,
    )


    checkpoint = torch.load(
        FLOW_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )


    state_dict = (
        _extract_state_dict(
            checkpoint
        )
    )


    cleaned = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value


    # --------------------------------------------------------
    # Load.
    #
    # strict=True is intentional.
    #
    # We do NOT want to silently run a different architecture
    # against your frozen checkpoint.
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned,
            strict=True,
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "\nRealNVP checkpoint does not "
            "match the reconstructed architecture.\n"
            "This is important — DO NOT continue "
            "with a partially loaded flow.\n\n"
            f"{exc}"
        ) from exc


    model.to(
        DEVICE
    )

    model.eval()


    # --------------------------------------------------------
    # Saved latent normalization.
    #
    # These are the values fitted during the original
    # RealNVP training pipeline.
    # --------------------------------------------------------

    latent_mean = torch.load(
        LATENT_MEAN_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    latent_std = torch.load(
        LATENT_STD_PATH,
        map_location=DEVICE,
        weights_only=False,
    )


    latent_mean = torch.as_tensor(
        latent_mean,
        dtype=torch.float32,
        device=DEVICE,
    ).flatten()


    latent_std = torch.as_tensor(
        latent_std,
        dtype=torch.float32,
        device=DEVICE,
    ).flatten()


    if (
        latent_mean.numel()
        != LATENT_DIM
    ):

        raise RuntimeError(
            "latent_mean.pt does not contain "
            f"{LATENT_DIM} values."
        )


    if (
        latent_std.numel()
        != LATENT_DIM
    ):

        raise RuntimeError(
            "latent_std.pt does not contain "
            f"{LATENT_DIM} values."
        )


    latent_std = torch.clamp(
        latent_std,
        min=1e-8,
    )


    print(
        "RealNVP loaded successfully."
    )

    print(
        f"  checkpoint : {FLOW_CHECKPOINT}"
    )

    print(
        f"  latent dim : {LATENT_DIM}"
    )

    print(
        f"  layers     : {NUM_LAYERS}"
    )

    print(
        f"  hidden     : {HIDDEN_DIM}"
    )

    print(
        f"  clamp      : {SCALE_CLAMP}"
    )

    print(
        f"  parameters : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )


    return (
        model,
        latent_mean,
        latent_std,
    )


# ============================================================
# SCORE A VAE LATENT
# ============================================================

@torch.inference_mode()
def score_latent(
    model: RealNVP,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    mu: torch.Tensor,
) -> float:

    if mu.ndim == 1:

        mu = mu.unsqueeze(0)


    mu = mu.to(
        DEVICE,
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # EXACT normalization used for the flow.
    # --------------------------------------------------------

    standardized = (
        mu - latent_mean
    ) / latent_std


    nll = flow_nll(
        model,
        standardized,
    )


    return float(
        nll[0]
        .detach()
        .cpu()
        .item()
    )