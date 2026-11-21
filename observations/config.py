import copy
import json


def _normalize_dataset(entry):
    if isinstance(entry, str):
        return {"name": entry, "roi": {}}
    norm = copy.deepcopy(entry)
    norm.setdefault("roi", {})
    return norm


def _normalize_method(entry):
    if isinstance(entry, str):
        return {"name": entry}
    return copy.deepcopy(entry)


def load_config(path):
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    cfg['datasets'] = [_normalize_dataset(d) for d in cfg['datasets']]
    cfg['methods'] = [_normalize_method(m) for m in cfg['methods']]

    osc_cfg = copy.deepcopy(cfg.get('oscillator', {}))
    osc_cfg.setdefault('fs', 64.0)
    osc_cfg.setdefault('f_min', 0.08)
    osc_cfg.setdefault('f_max', 0.5)
    osc_cfg.setdefault('tau_env', 30.0)
    osc_cfg.setdefault('qx_scale', 0.3)
    osc_cfg.setdefault('rv_auto', True)
    osc_cfg.setdefault('rv_mad_scale', 1.2)
    osc_cfg.setdefault('rv_floor', 0.08)
    osc_cfg.setdefault('qf', 5e-5)
    osc_cfg.setdefault('stft_win', 12.0)
    osc_cfg.setdefault('stft_hop', 1.0)
    osc_cfg.setdefault('ridge_penalty', 250.0)
    osc_cfg.setdefault('pll_autogain', True)
    osc_cfg.setdefault('pll_zeta', 0.9)
    osc_cfg.setdefault('pll_ttrack', 7.0)
    cfg['oscillator'] = osc_cfg

    return cfg
