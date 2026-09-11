"""
QMIX mixing network (Rashid et al., 2018): combines a team's N_TEAM per-agent chosen-action
Q-values into one team-level Q_tot, conditioned on a global state embedding, under a
monotonicity constraint (dQ_tot/dQ_i >= 0 for every agent i).

Monotonicity is enforced by using only non-negative mixing weights (abs() of a
hypernetwork's output) — it's what lets greedy argmax on Q_tot factor into independent
per-agent argmax on each Q_i, so units can still act by locally maximizing their own Q row
(see MyPolicy.act()) even though training optimizes the team-level Q_tot. QMIX only changes
how the *training target* is computed, not how actions are selected at inference time.

state is MyModel's global_latent (the attention-refined team_state token) rather than a
separate hand-built state vector — it's already a summary that attended over every
vision/unit token in the trunk, so there's no reason to build a second, cruder one here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 64,
        hyper_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agent_qs : [B, n_agents]  chosen-action Q-value per teammate
        state    : [B, state_dim] global state embedding (MyModel's global_latent)

        Returns
        -------
        q_tot : [B, 1]
        """
        B = agent_qs.shape[0]

        w1 = self.hyper_w1(state).abs().view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(B, 1, self.n_agents), w1) + b1)  # [B, 1, embed_dim]

        w2 = self.hyper_w2(state).abs().view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # [B, 1, 1]

        return q_tot.view(B, 1)


class DistributionalQMixer(nn.Module):
    """
    Distributional (DFAC, Sun et al. 2021) extension of QMixer for combining IQN's per-agent
    quantile functions into a team quantile function, WITHOUT breaking QMIX's IGM guarantee.

    An earlier version of this reused QMixer's full nonlinear forward once per quantile
    sample (Z_tot(tau) = QMixer(Z_1(tau),...,Zn(tau), state) for every tau). That's wrong:
    QMixer has an ELU nonlinearity, so E_tau[QMixer(Z_1(tau),...)] != QMixer(E[Z_1],...) in
    general — the team value actually being *optimized* (an average over quantile samples of
    a nonlinear mixture) would then disagree with the value implied by each agent picking its
    own greedy action by mean-Q (which is what action selection / the Double-DQN "which
    action is greedy" step actually does). That's IGM (Individual-Global-Max) breaking at the
    mean level — the whole reason QMIX's mixing weights are constrained non-negative.

    The fix: split each agent's quantile function into its mean and a zero-mean "shape",
    mix the MEANS through the ordinary nonlinear QMixer (exactly recovering scalar QMIX and
    its IGM guarantee), and mix the zero-mean shapes through a separate, strictly
    non-negative *linear* combination — linear so the team shape has exactly zero mean, which
    is what makes the whole construction's mean come out exactly equal to the scalar QMIX
    value by construction, not by accident:

        Z_tot(tau) = QMixer(E[Z_1],...,E[Zn], state) + sum_i w_i(state) * (Z_i(tau) - E[Zi])
        w_i(state) >= 0

    so mean_tau[Z_tot(tau)] = QMixer(E[Z_1],...,E[Zn], state) exactly, for any w_i.
    """

    def __init__(self, mixer: QMixer, state_dim: int, n_agents: int, hyper_hidden: int = 128) -> None:
        super().__init__()
        self.mixer = mixer
        self.shape_weight = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents),
        )

    def forward(self, chosen_quantiles: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        chosen_quantiles : [B, n_agents, Q]  each teammate's chosen-action quantile values
        state            : [B, state_dim]    tau-independent (MyModel's global_latent)

        Returns
        -------
        Z_tot : [B, Q]  team quantile function, evaluated at the same Q quantile samples
        """
        mean_q = chosen_quantiles.mean(dim=2)  # [B, n_agents] -- exactly what q_values already is
        deviation = chosen_quantiles - mean_q.unsqueeze(2)  # [B, n_agents, Q], zero-mean per agent

        q_tot_mean = self.mixer(mean_q, state)  # [B, 1] -- ordinary scalar QMIX, IGM intact
        w = self.shape_weight(state).abs()  # [B, n_agents], non-negative -> linear, mean-preserving
        shape_tot = torch.einsum("baq,ba->bq", deviation, w)  # [B, Q]

        return q_tot_mean + shape_tot  # [B, Q], q_tot_mean broadcasts over the quantile axis
