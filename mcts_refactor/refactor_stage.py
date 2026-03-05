from collections import Counter, defaultdict


def clean_pipelines(pipe_configs, config_pairs, need_imv, need_o, need_e):
    """Stage 2 (part 1): clean historical pipeline candidates with meta constraints."""
    cleaned_pipe_configs = []
    for cfg in pipe_configs:
        new_cfg = cfg.copy()
        if not need_imv and "I" in new_cfg:
            del new_cfg["I"]
        if not need_o and "O" in new_cfg:
            del new_cfg["O"]
        if not need_e and "E" in new_cfg:
            del new_cfg["E"]
        if "FS" in new_cfg:
            del new_cfg["FS"]
        cleaned_pipe_configs.append(new_cfg)

    cleaned_config_pairs = []
    for order, config, source in config_pairs:
        new_order = []
        new_config = []
        for name, value in zip(order, config[1:]):
            if name == "O" and not need_o:
                continue
            if name == "E" and not need_e:
                continue
            if name == "I" and not need_imv:
                continue
            if name == "FS":
                continue
            new_order.append(name)
            new_config.append(value)

        if need_imv:
            cleaned_config_pairs.append((["I"] + new_order, [config[0]] + new_config, source))
        else:
            cleaned_config_pairs.append((new_order, new_config, source))

    merged = {}
    for order, _config, source in cleaned_config_pairs:
        key = tuple(order)
        if key not in merged:
            merged[key] = set(source)
        else:
            merged[key].update(source)
    unique_logic = {logic: tuple(sorted(src)) for logic, src in merged.items()}

    unique_dicts = [
        dict(t)
        for t in set(
            tuple((k, v) for k, v in d.items() if k != "source_dataset") for d in cleaned_pipe_configs
        )
    ]
    unique_config_pairs = [(list(d.keys()), list(d.values())) for d in unique_dicts]

    return unique_dicts, unique_config_pairs, unique_logic


def build_candidate_configs(config_pair_cleaned):
    """Stage 2 (part 2): build candidate logic pipelines with default physical ops."""
    pipe_op = defaultdict(int)
    component_op_freq = defaultdict(Counter)

    max_length = max(len(order) for order, _ in config_pair_cleaned)
    filled_config_pair = []
    for order, ops in config_pair_cleaned:
        order_filled = order + [None] * (max_length - len(order))
        ops_filled = ops + [None] * (max_length - len(ops))
        filled_config_pair.append((order_filled, ops_filled))

    for order, ops in filled_config_pair:
        pipe_op[tuple(order)] += 1
        for component, operation in zip(order, ops):
            if operation is not None:
                component_op_freq[component][operation] += 1

    candidate_pipe = [list(d) for d in pipe_op]
    def_ops = {
        component: max(op_freq.items(), key=lambda x: x[1])[0]
        for component, op_freq in component_op_freq.items()
    }

    candidate_config_pair = []
    for tmp in candidate_pipe:
        ops = [def_ops[tf] for tf in tmp]
        candidate_config_pair.append((tmp, ops))

    return candidate_config_pair, {
        "candidate_logic_count": len(candidate_config_pair),
        "candidate_pipelines": [{"order": cp[0], "default_ops": cp[1]} for cp in candidate_config_pair],
    }


def refactor_candidates(pipe_configs, config_pair, need_imv, need_o, need_e):
    """Stage 2: clean + deduplicate + build candidate pipelines."""
    _, config_pair_cleaned, unique_logic = clean_pipelines(
        pipe_configs, config_pair, need_imv, need_o, need_e
    )
    candidate_config_pair, candidate_stats = build_candidate_configs(config_pair_cleaned)

    return {
        "config_pair_cleaned": config_pair_cleaned,
        "unique_logic": unique_logic,
        "candidate_config_pair": candidate_config_pair,
        "stage2_stats": {
            "cleaned_logic_count": len(config_pair_cleaned),
            "unique_logic_sources": {
                "_".join(k): [str(vv) for vv in v] for k, v in unique_logic.items()
            },
        },
        "stage1_candidate_stats": candidate_stats,
    }
