objectives = ax_client.experiment.optimization_config.objective.objectives
all_metrics=[objectives[x].metric for x in range(0, len(objectives))]
all_keys=[vars(x)["_name"] for x in all_metrics]
metric_dict=dict(zip(all_keys, all_metrics))
combinations=list(itertools.combinations(all_keys, 2))

