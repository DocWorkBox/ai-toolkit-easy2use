import torch
from typing import List


class Singularity(torch.optim.Optimizer):
    """
    Experimental memory-lean optimizer for diffusion LoRA training.

    Singularity combines Adafactor-style factored second moments with per-row
    learning rates. It stores one previous preconditioned update instead of
    Automagic's polarity history list, then uses row-wise cosine agreement to
    nudge learning rates geometrically.
    """

    _DTYPES = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        min_lr: float = 1e-8,
        max_lr: float = 3e-3,
        lr_bump_rate: float = 0.05,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        target_update_ratio: float = 0.01,
        cosine_floor: float = 0.0,
        state_dtype: str = "auto",
        fused: bool = False,
        stochastic_accumulation: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if min_lr <= 0.0 or max_lr <= 0.0 or min_lr > max_lr:
            raise ValueError(f"Invalid lr bounds: min_lr={min_lr}, max_lr={max_lr}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if target_update_ratio < 0.0:
            raise ValueError(f"Invalid target_update_ratio: {target_update_ratio}")
        if not -1.0 <= cosine_floor < 1.0:
            raise ValueError(f"Invalid cosine_floor: {cosine_floor}")

        defaults = dict(
            lr=float(min(max(lr, min_lr), max_lr)),
            min_lr=float(min_lr),
            max_lr=float(max_lr),
            lr_bump_rate=float(lr_bump_rate),
            beta2=float(beta2),
            eps=float(eps),
            clip_threshold=float(clip_threshold),
            weight_decay=float(weight_decay),
            target_update_ratio=float(target_update_ratio),
            cosine_floor=float(cosine_floor),
            state_dtype=state_dtype,
            stochastic_accumulation=bool(stochastic_accumulation),
        )
        super().__init__(params, defaults)

        self.fused = fused
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    self._hook_handles.append(
                        p.register_post_accumulate_grad_hook(
                            self._make_backward_hook(group)
                        )
                    )
                elif group["stochastic_accumulation"] and p.dtype != torch.float32:
                    self._hook_handles.append(
                        p.register_post_accumulate_grad_hook(
                            self._make_accum_hook()
                        )
                    )

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        if not v_fp32.is_contiguous():
            v_fp32 = v_fp32.contiguous()
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    def _state_dtype_for(self, p: torch.Tensor, group: dict) -> torch.dtype:
        requested = str(group["state_dtype"]).lower()
        if requested == "param":
            return p.dtype if p.dtype.is_floating_point else torch.float32
        if requested == "auto":
            if p.dtype in (torch.float16, torch.bfloat16):
                return p.dtype
            return torch.float16 if p.device.type == "cuda" else torch.float32
        if requested not in self._DTYPES:
            raise ValueError(f"Unknown Singularity state_dtype: {group['state_dtype']}")
        return self._DTYPES[requested]

    def _make_accum_hook(self):
        def _hook(p: torch.Tensor):
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state_dtype = self._state_dtype_for(p, group)
        lr_shape = (p.shape[0],) if p.dim() >= 2 else p.shape
        state["step"] = 0
        state["lr"] = torch.full(
            lr_shape, float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["prev_update"] = torch.zeros_like(p, dtype=state_dtype)
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=state_dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=state_dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(
                p.shape, dtype=state_dtype, device=p.device
            )

    def _row_view(self, lr_t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if p.dim() >= 2:
            return lr_t.view(lr_t.shape[0], *([1] * (p.dim() - 1)))
        return lr_t

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        if p.grad.is_sparse:
            raise RuntimeError("Singularity does not support sparse gradients.")

        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            row = row_state.to(torch.float32)
            col = col_state.to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            row_state.copy_(row.to(row_state.dtype))
            col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
            reduce_dims = tuple(range(1, p.dim()))
        else:
            v_state = state["exp_avg_sq"]
            v = v_state.to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            v_state.copy_(v.to(v_state.dtype))
            update = v.add(eps).rsqrt().mul_(grad)
            reduce_dims = None

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        lr_t = state["lr"]
        lr_b = self._row_view(lr_t, p)
        prev_update = state["prev_update"].to(torch.float32)

        if state["step"] > 0:
            if reduce_dims is None:
                denom = update.abs().mul(prev_update.abs()).add_(eps)
                cosine = update.mul(prev_update).div_(denom).clamp_(-1.0, 1.0)
            else:
                dot = update.mul(prev_update).mean(dim=reduce_dims)
                cur = update.square().mean(dim=reduce_dims).sqrt_()
                prev = prev_update.square().mean(dim=reduce_dims).sqrt_()
                cosine = dot.div_(cur.mul_(prev).add_(eps)).clamp_(-1.0, 1.0)
            floor = group["cosine_floor"]
            direction = cosine.sub_(floor).div_(1.0 - floor).clamp_(-1.0, 1.0)
            lr_t.mul_(torch.exp(direction.mul_(group["lr_bump_rate"]))).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )

        state["prev_update"].copy_(update.to(state["prev_update"].dtype))
        state["step"] += 1

        p_fp32 = p if p.dtype == torch.float32 else p.to(torch.float32)
        if group["weight_decay"] != 0.0:
            update.add_(p_fp32, alpha=group["weight_decay"])
        update.mul_(lr_b)

        if group["target_update_ratio"] > 0.0:
            floor = 1e-3
            if reduce_dims is None:
                param_scale = p_fp32.abs().clamp_min(floor)
                cap = param_scale.mul(group["target_update_ratio"])
                scale = cap.div(update.abs().add(eps)).clamp_(max=1.0)
                update.mul_(scale)
            else:
                delta_rms = update.square().mean(dim=reduce_dims).sqrt_()
                param_rms = p_fp32.square().mean(dim=reduce_dims).sqrt_().clamp_min_(floor)
                cap = param_rms.mul_(group["target_update_ratio"])
                scale = cap.div_(delta_rms.add_(eps)).clamp_(max=1.0)
                update.mul_(self._row_view(scale, p))

        if p.dtype == torch.float32:
            p.add_(update, alpha=-1.0)
        else:
            new_p_fp32 = p.to(torch.float32).add_(update, alpha=-1.0)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.fused:
            return loss

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                accum = getattr(p, "_accum_grad", None)
                if accum is not None:
                    p.grad = accum
                    del p._accum_grad
                self._update_param(p, group)
        return loss

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                float(self.state[p]["lr"].mean())
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(sum(lrs) / len(lrs) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                state_dtype = self._state_dtype_for(p, group)
                for key in ("prev_update", "exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col"):
                    if isinstance(st.get(key), torch.Tensor):
                        st[key] = st[key].to(dtype=state_dtype, device=p.device)
