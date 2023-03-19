# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True):
    """Register all modules in mmdet into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmdet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmyolo`, and all registries will build modules from mmdet's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmdet.engine  # noqa: F401,F403
    import mmdet.visualization  # noqa: F401,F403

    import mmyolo.datasets  # noqa: F401,F403
    import mmyolo.engine  # noqa: F401,F403
    import mmyolo.models  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmyolo')
        if never_created:
            DefaultScope.get_instance('mmyolo', scope_name='mmyolo')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmyolo':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmyolo", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmyolo". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmyolo-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmyolo')
