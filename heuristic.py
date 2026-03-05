def update_heuristic_table(
    action_score_table: dict,
    current_config: list,
    score: float,
    best_score: float,
    start_token: str = "__START__",
    allow_start_negative: bool = False,
    allow_negative: bool = True,
    alpha: float = 1.0,
):
    """
    Unified heuristic score-table update with explicit START handling.

    Parameters
    ----------
    action_score_table : dict
        Heuristic table in the form {prev_action: {next_action: score}}.

    current_config : list
        Current full physical configuration, e.g. [a0, a1, a2, ...].

    score : float
        Evaluation score of current_config.

    best_score : float
        Best score seen so far (used to compute delta).

    start_token : str
        Start-state token (default "__START__").

    allow_start_negative : bool
        Whether to allow negative updates for START -> first_action.

    allow_negative : bool
        Whether to allow negative updates for intermediate transitions.

    alpha : float
        Update step size (learning rate).
    """
    if not current_config:
        return

    dlter = score - best_score

    first_action = current_config[0]
    action_score_table.setdefault(start_token, {})

    if dlter > 0 or allow_start_negative:
        action_score_table[start_token][first_action] = (
            action_score_table[start_token].get(first_action, 0.0)
            + alpha * dlter
        )

    for prev, nxt in zip(current_config[:-1], current_config[1:]):
        action_score_table.setdefault(prev, {})

        if dlter > 0 or allow_negative:
            action_score_table[prev][nxt] = (
                action_score_table[prev].get(nxt, 0.0)
                + alpha * dlter
            )
