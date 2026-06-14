"""WAN 2.2 model family (T2V-A14B / I2V-A14B — dual-expert MoE).

WAN 2.2 A14B is a Mixture-of-Experts diffusion DiT: a **high-noise expert**
(``transformer``, active for ``t >= boundary``) and a **low-noise expert**
(``transformer_2``, active for ``t < boundary``). This family trains BOTH
experts in a single run by auto-switching the active expert per optimizer step
(routed by the sampled timestep) and emits TWO LoRA files (high + low).

All the shared WAN flow-match / forward / sampler logic is reused from
``wan_shared`` (the WAN 2.1 phase). This package adds only the dual-expert
router, dual-adapter trainer/saver, and the swap-aware sampler.
"""
