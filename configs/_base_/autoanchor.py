k_means_autoanchor_hook = dict(
    type='YOLOAutoAnchorHook',
    optimizer=dict(
        type='YOLOKMeansAnchorOptimizer',
        iters=1000,
        num_anchor_per_level=[3, 3, 3]))

de_autoanchor_hook = dict(
    type='YOLOAutoAnchorHook',
    optimizer=dict(
        type='YOLODEAnchorOptimizer',
        iters=1000,
        num_anchor_per_level=[3, 3, 3]))

v5_k_means_autoanchor_hook = dict(
    type='YOLOAutoAnchorHook',
    optimizer=dict(
        type='YOLOV5KMeansAnchorOptimizer',
        iters=1000,
        num_anchor_per_level=[3, 3, 3],
        prior_match_thr=4.0,
        mutation_args=[0.9, 0.1],
        augment_args=[0.9, 0.1]))
