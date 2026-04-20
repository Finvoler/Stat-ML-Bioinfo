"""
Survival analysis: Cox Proportional Hazards model and DeepSurv.

Survival data has a unique structure: right-censored observations where
we only know a patient survived BEYOND time T (event did not occur by
study end). Standard regression (MSE) is invalid for censored outcomes.

Cox PH Model (Cox, 1972):
    h(t|x) = h_0(t) * exp(β^T x)
    Linear predictor — cannot capture gene-gene interaction effects.

DeepSurv (Katzman et al., 2018):
    h(t|x) = h_0(t) * exp(f_θ(x))
    f_θ is a deep neural network, learning arbitrary non-linear risk functions.
    Loss = negative log partial likelihood + L2 regularization on θ.

Both models are evaluated by C-index (concordance index):
    C = P(r_i > r_j | T_i < T_j)  — fraction of correctly ordered risk pairs.
    C=0.5: random; C=1.0: perfect.

References
----------
Cox (1972). Regression models and life-tables. JRSS-B, 34(2), 187-202.
Katzman et al. (2018). DeepSurv. BMC Medical Research Methodology, 18(1), 1-12.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


# ---------------------------------------------------------------------------
# Cox Proportional Hazards wrapper (via lifelines)
# ---------------------------------------------------------------------------

class CoxPHWrapper:
    """
    Scikit-learn-style wrapper around lifelines.CoxPHFitter.

    Parameters
    ----------
    penalizer : float
        L2 regularization strength. Equivalent to Ridge-penalized Cox,
        helpful when feature count approaches sample count.
    duration_col : str
        Column name for survival time in the input DataFrame.
    event_col : str
        Column name for event indicator (1=event, 0=censored).
    """

    def __init__(
        self,
        penalizer: float = 0.1,
        duration_col: str = "duration",
        event_col: str = "event",
    ):
        self.penalizer = penalizer
        self.duration_col = duration_col
        self.event_col = event_col
        self._fitter = CoxPHFitter(penalizer=penalizer)

    def fit(
        self,
        X: pd.DataFrame,
        durations: pd.Series,
        events: pd.Series,
    ) -> "CoxPHWrapper":
        """
        Fit the Cox model.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix (samples x features).
        durations : pd.Series
            Observed survival times.
        events : pd.Series
            Event indicators (1=occurred, 0=censored).
        """
        df = X.copy()
        df[self.duration_col] = durations.values
        df[self.event_col] = events.values
        self._fitter.fit(df, duration_col=self.duration_col, event_col=self.event_col)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Return partial hazard (exp(β^T x)) as risk scores."""
        return self._fitter.predict_partial_hazard(X).values

    def c_index(
        self,
        X: pd.DataFrame,
        durations: pd.Series,
        events: pd.Series,
    ) -> float:
        """Compute C-index on held-out data."""
        risk = self.predict_risk(X)
        return concordance_index(durations, -risk, events)

    def summary(self) -> pd.DataFrame:
        """Return model summary with hazard ratios, p-values, and CIs."""
        return self._fitter.summary


# ---------------------------------------------------------------------------
# DeepSurv neural network (PyTorch)
# ---------------------------------------------------------------------------

class DeepSurv(nn.Module):
    """
    DeepSurv: Cox PH with deep neural network risk function.

    Replaces the linear β^T x predictor with a multi-layer network f_θ(x):

        h(t|x) = h_0(t) · exp(f_θ(x))

    Training minimizes the negative log partial likelihood:

        L(θ) = -Σ_{i∈E} [f_θ(x_i) - log Σ_{j∈R(t_i)} exp(f_θ(x_j))] + λ||θ||²

    This allows the model to capture complex non-linear risk patterns such as
    synergistic gene expression effects that Cox cannot represent.

    Parameters
    ----------
    in_features : int
        Number of input features (genes).
    hidden_layers : list of int
        Sizes of hidden layers. Default [256, 128, 64].
    dropout : float
        Dropout rate for regularization. Default 0.4.
    batch_norm : bool
        Whether to apply batch normalization after each layer. Default True.
    activation : str
        Activation function: 'relu' or 'selu'. Default 'relu'.
    """

    def __init__(
        self,
        in_features: int,
        hidden_layers: list = None,
        dropout: float = 0.4,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_layers = hidden_layers or [256, 128, 64]

        act_fn = {"relu": nn.ReLU, "selu": nn.SELU, "tanh": nn.Tanh}[activation]

        layers = []
        prev = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev = h
        # Single output unit: log-risk score (no activation — follows Cox convention)
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def cox_partial_likelihood_loss(
        log_risk: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log partial likelihood.

        The risk set R(t_i) consists of all subjects who have not yet
        experienced the event at time t_i. Sorting by duration (descending)
        allows efficient computation via cumulative log-sum-exp.

        Parameters
        ----------
        log_risk : torch.Tensor, shape (n,)
            Network output f_θ(x).
        durations : torch.Tensor, shape (n,)
            Observed survival times.
        events : torch.Tensor, shape (n,)
            Event indicators (1.0 = event, 0.0 = censored).
        """
        # Sort by descending duration so cumsum acts as risk-set aggregator
        order = torch.argsort(durations, descending=True)
        log_risk = log_risk[order]
        events = events[order]

        # log Σ_{j∈R(t_i)} exp(f_θ(x_j)) using numerically stable cumlogsumexp
        log_cumsum = torch.logcumsumexp(log_risk, dim=0)

        # Sum only over event-experiencing subjects (censored contribute 0)
        loss = -((log_risk - log_cumsum) * events).sum() / (events.sum() + 1e-8)
        return loss


class DeepSurvTrainer:
    """
    Training loop for DeepSurv with learning rate scheduling and early stopping.

    Parameters
    ----------
    model : DeepSurv
        Initialized DeepSurv network.
    lr : float
        Initial learning rate. Default 1e-3.
    weight_decay : float
        L2 regularization on network parameters λ. Default 1e-4.
    epochs : int
        Maximum training epochs. Default 300.
    patience : int
        Early stopping patience (epochs without C-index improvement). Default 30.
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    """

    def __init__(
        self,
        model: DeepSurv,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 300,
        patience: int = 30,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.patience = patience
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        self.history = {"train_loss": [], "val_cindex": []}

    def fit(
        self,
        X_train: np.ndarray,
        durations_train: np.ndarray,
        events_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        durations_val: Optional[np.ndarray] = None,
        events_val: Optional[np.ndarray] = None,
    ) -> "DeepSurvTrainer":
        """Train DeepSurv with optional validation-based early stopping."""
        X_t = torch.FloatTensor(X_train).to(self.device)
        dur_t = torch.FloatTensor(durations_train).to(self.device)
        evt_t = torch.FloatTensor(events_train).to(self.device)

        best_cindex = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            log_risk = self.model(X_t)
            loss = DeepSurv.cox_partial_likelihood_loss(log_risk, dur_t, evt_t)
            loss.backward()
            self.optimizer.step()
            self.history["train_loss"].append(loss.item())

            # Validation C-index
            if X_val is not None:
                c = self.c_index(X_val, durations_val, events_val)
                self.history["val_cindex"].append(c)
                self.scheduler.step(c)

                if c > best_cindex:
                    best_cindex = c
                    patience_counter = 0
                    self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.model.load_state_dict(self._best_state)
                        break

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return log-risk scores f_θ(x) for each sample."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            return self.model(X_t).cpu().numpy()

    def c_index(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """Compute Harrell's C-index on the given split."""
        risk = self.predict_risk(X)
        return concordance_index(durations, -risk, events)
