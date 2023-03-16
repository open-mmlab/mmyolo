autoanchor_hook = dict(
    type='YOLOAutoAnchorHook',
    optimizer=dict(
        type='YOLOKMeansAnchorOptimizer',
        iters=1000,
        num_anchor_per_level=[3, 3, 3]))
