# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import os
import os.path as osp
import pkgutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# host_addr = 'https://gitee.com/open-mmlab'
host_addr = 'https://github.com/open-mmlab'
tools_list = ['tools', '.dev_scripts']
proxy_names = {
    'mmdet': 'mmdetection',
    'mmseg': 'mmsegmentation',
    'mmcls': 'mmclassification'
}
merge_module_keys = {'mmcv': ['mmengine']}
# exclude_prefix = {'mmcv': ['<class \'mmengine.model.']}
exclude_prefix = {}
markdown_title = '# MM 系列开源库注册表\n'
markdown_title += '（注意：本文档是通过 .dev_scripts/print_registers.py 脚本自动生成）'


def capitalize(repo_name):
    lower = repo_name.lower()
    if lower == 'mmcv':
        return repo_name.upper()
    elif lower.startswith('mm'):
        return 'MM' + repo_name[2:]
    return repo_name.capitalize()


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def parse_repo_name(repo_name):
    proxy_names_rev = dict(zip(proxy_names.values(), proxy_names.keys()))
    repo_name = proxy_names.get(repo_name, repo_name)
    module_name = proxy_names_rev.get(repo_name, repo_name)
    return repo_name, module_name


def git_pull_branch(repo_name, branch_name='', pulldir='.'):
    mkdir_or_exist(pulldir)
    exec_str = f'cd {pulldir};git init;git pull '
    exec_str += f'{host_addr}/{repo_name}.git'
    if branch_name:
        exec_str += f' {branch_name}'
    returncode = os.system(exec_str)
    if returncode:
        raise RuntimeError(
            f'failed to get the remote repo, code: {returncode}')


def load_modules_from_dir(module_name, module_root, throw_error=False):
    print(f'loading the {module_name} modules...')
    # # install the dependencies
    # if osp.exists(osp.join(pkg_dir, 'requirements.txt')):
    #     os.system('pip install -r requirements.txt')
    # get all module list
    module_list = []
    error_dict = {}
    module_root = osp.join(module_root, module_name)
    assert osp.exists(module_root), \
        f'cannot find the module root: {module_root}'
    for _root, _dirs, _files in os.walk(module_root):
        if (('__init__.py' not in _files)
                and (osp.split(_root)[1] != '__pycache__')):
            # add __init__.py file to the package
            with open(osp.join(_root, '__init__.py'), 'w') as _:
                pass

    def _onerror(*args, **kwargs):
        pass

    for _finder, _name, _ispkg in pkgutil.walk_packages([module_root],
                                                        prefix=module_name +
                                                        '.',
                                                        onerror=_onerror):
        try:
            module = importlib.import_module(_name)
            module_list.append(module)
        except Exception as e:
            if throw_error:
                raise e
            _error_msg = f'{type(e)}: {e}.'
            print(f'cannot import the module: {_name} ({_error_msg})')
            assert (_name not in error_dict), \
                f'duplicate error name was found: {_name}'
            error_dict[_name] = _error_msg
    for module in module_list:
        assert module.__file__.startswith(module_root), \
            f'the importing path of package was wrong: {module.__file__}'
    print('modules were loaded...')
    return module_list, error_dict


def get_registries_from_modules(module_list):
    registries = {}
    objects_set = set()
    # import the Registry class,
    # import at the beginning is not allowed
    # because it is not the temp package
    from mmengine.registry import Registry

    # only get the specific registries in module list
    for module in module_list:
        for obj_name in dir(module):
            _obj = getattr(module, obj_name)
            if isinstance(_obj, Registry):
                objects_set.add(_obj)
    for _obj in objects_set:
        if _obj.scope not in registries:
            registries[_obj.scope] = {}
        registries_scope = registries[_obj.scope]
        assert _obj.name not in registries_scope, \
            f'multiple definition of {_obj.name} in registries'
        registries_scope[_obj.name] = {
            key: str(val)
            for key, val in _obj.module_dict.items()
        }
    print('registries got...')
    return registries


def merge_registries(src_dict, dst_dict):
    assert type(src_dict) == type(dst_dict), \
        (f'merge type is not supported: '
         f'{type(dst_dict)} and {type(src_dict)}')
    if isinstance(src_dict, str):
        return
    for _k, _v in dst_dict.items():
        if (_k not in src_dict):
            src_dict.update({_k: _v})
        else:
            assert isinstance(_v, (dict, str)) and \
                isinstance(src_dict[_k], (dict, str)), \
                'merge type is not supported: ' \
                f'{type(_v)} and {type(src_dict[_k])}'
            merge_registries(src_dict[_k], _v)


def exclude_registries(registries, exclude_key):
    for _k in list(registries.keys()):
        _v = registries[_k]
        if isinstance(_v, str) and _v.startswith(exclude_key):
            registries.pop(_k)
        elif isinstance(_v, dict):
            exclude_registries(_v, exclude_key)


def get_scripts_from_dir(root):

    def _recurse(_dict, _chain):
        if len(_chain) <= 1:
            _dict[_chain[0]] = None
            return
        _key, *_chain = _chain
        if _key not in _dict:
            _dict[_key] = {}
        _recurse(_dict[_key], _chain)

    # find all scripts in the root directory. (not just ('.py', '.sh'))
    # can not use the scandir function in mmengine to scan the dir,
    # because mmengine import is not allowed before git pull
    scripts = {}
    for _subroot, _dirs, _files in os.walk(root):
        for _file in _files:
            _script = osp.join(osp.relpath(_subroot, root), _file)
            _recurse(scripts, Path(_script).parts)
    return scripts


def get_version_from_module_name(module_name, branch):
    branch_str = str(branch) if branch is not None else ''
    version_str = ''
    try:
        exec(f'import {module_name}')
        _module = eval(f'{module_name}')
        if hasattr(_module, '__version__'):
            version_str = str(_module.__version__)
        else:
            version_str = branch_str
        version_str = f' ({version_str})' if version_str else version_str
    except (ImportError, AttributeError) as e:
        print(f'can not get the version of module {module_name}: {e}')
    return version_str


def print_tree(print_dict):
    # recursive print the dict tree
    def _recurse(_dict, _connector='', n=0):
        assert isinstance(_dict, dict), 'recursive type must be dict'
        tree = ''
        for idx, (_key, _val) in enumerate(_dict.items()):
            sub_tree = ''
            _last = (idx == (len(_dict) - 1))
            if isinstance(_val, str):
                _key += f' ({_val})'
            elif isinstance(_val, dict):
                sub_tree = _recurse(_val,
                                    _connector + ('   ' if _last else '│  '),
                                    n + 1)
            else:
                assert (_val is None), f'unknown print type {_val}'
            tree += '  ' + _connector + \
                    ('└─' if _last else '├─') + f'({n}) {_key}' + '\n'
            tree += sub_tree
        return tree

    for _pname, _pdict in print_dict.items():
        print('-' * 100)
        print(f'{_pname}\n' + _recurse(_pdict))


def divide_list_into_groups(_array, _maxsize_per_group):
    if not _array:
        return _array
    _groups = np.asarray(len(_array) / _maxsize_per_group)
    if len(_array) % _maxsize_per_group:
        _groups = np.floor(_groups) + 1
    _groups = _groups.astype(int)
    return np.array_split(_array, _groups)


def registries_to_html(registries, title=''):
    max_col_per_row = 5
    max_size_per_cell = 20
    html = ''
    table_data = []
    # save repository registries
    for registry_name, registry_dict in registries.items():
        # filter the empty registries
        if not registry_dict:
            continue
        registry_strings = []
        if isinstance(registry_dict, dict):
            registry_dict = list(registry_dict.keys())
        elif isinstance(registry_dict, list):
            pass
        else:
            raise TypeError(
                f'unknown type of registry_dict {type(registry_dict)}')
        for _k in registry_dict:
            registry_strings.append(f'<li>{_k}</li>')
        table_data.append((registry_name, registry_strings))

    # sort the data list
    table_data = sorted(table_data, key=lambda x: len(x[1]))
    # split multi parts
    table_data_multi_parts = []
    for (registry_name, registry_strings) in table_data:
        multi_parts = False
        if len(registry_strings) > max_size_per_cell:
            multi_parts = True
        for cell_idx, registry_cell in enumerate(
                divide_list_into_groups(registry_strings, max_size_per_cell)):
            registry_str = ''.join(registry_cell.tolist())
            registry_str = f'<ul>{registry_str}</ul>'
            table_data_multi_parts.append([
                registry_name if not multi_parts else
                f'{registry_name} (part {cell_idx + 1})', registry_str
            ])

    for table_data in divide_list_into_groups(table_data_multi_parts,
                                              max_col_per_row):
        table_data = list(zip(*table_data.tolist()))
        html += dataframe_to_html(
            pd.DataFrame([table_data[1]], columns=table_data[0]))
    if html:
        html = f'<div align=\'center\'><b>{title}</b></div>\n{html}'
        html = f'<details open>{html}</details>\n'
    return html


def tools_to_html(tools_dict, repo_name=''):

    def _recurse(_dict, _connector, _result):
        assert isinstance(_dict, dict), \
            f'unknown recurse type: {_dict} ({type(_dict)})'
        for _k, _v in _dict.items():
            if _v is None:
                if _connector not in _result:
                    _result[_connector] = []
                _result[_connector].append(_k)
            else:
                _recurse(_v, osp.join(_connector, _k), _result)

    table_data = {}
    title = f'{capitalize(repo_name)} Tools'
    _recurse(tools_dict, '', table_data)
    return registries_to_html(table_data, title)


def dataframe_to_html(dataframe):
    styler = dataframe.style
    styler = styler.hide(axis='index')
    styler = styler.format(na_rep='-')
    styler = styler.set_properties(**{
        'text-align': 'left',
        'align': 'center',
        'vertical-align': 'top'
    })
    styler = styler.set_table_styles([{
        'selector':
        'thead th',
        'props':
        'align:center;text-align:center;vertical-align:bottom'
    }])
    html = styler.to_html()
    html = f'<div align=\'center\'>\n{html}</div>'
    return html


def generate_markdown_by_repository(repo_name,
                                    module_name,
                                    branch,
                                    pulldir,
                                    throw_error=False):
    # add the pull dir to the system path so that it can be found
    if pulldir not in sys.path:
        sys.path.insert(0, pulldir)
    module_list, error_dict = load_modules_from_dir(
        module_name, pulldir, throw_error=throw_error)
    registries_tree = get_registries_from_modules(module_list)
    if error_dict:
        error_dict_name = 'error_modules'
        assert (error_dict_name not in registries_tree), \
            f'duplicate module name was found: {error_dict_name}'
        registries_tree.update({error_dict_name: error_dict})
    # get the tools files
    for tools_name in tools_list:
        assert (tools_name not in registries_tree), \
            f'duplicate tools name was found: {tools_name}'
        tools_tree = osp.join(pulldir, tools_name)
        tools_tree = get_scripts_from_dir(tools_tree)
        registries_tree.update({tools_name: tools_tree})
    # print_tree(registries_tree)
    # get registries markdown string
    module_registries = registries_tree.get(module_name, {})
    for merge_key in merge_module_keys.get(module_name, []):
        merge_dict = registries_tree.get(merge_key, {})
        merge_registries(module_registries, merge_dict)
    for exclude_key in exclude_prefix.get(module_name, []):
        exclude_registries(module_registries, exclude_key)
    markdown_str = registries_to_html(
        module_registries, title=f'{capitalize(repo_name)} Module Components')
    # get tools markdown string
    tools_registries = {}
    for tools_name in tools_list:
        tools_registries.update(
            {tools_name: registries_tree.get(tools_name, {})})
    markdown_str += tools_to_html(tools_registries, repo_name=repo_name)
    version_str = get_version_from_module_name(module_name, branch)
    title_str = f'\n\n## {capitalize(repo_name)}{version_str}\n'
    # remove the pull dir from system path
    if pulldir in sys.path:
        sys.path.remove(pulldir)
    return f'{title_str}{markdown_str}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='print registries in openmmlab repositories')
    parser.add_argument(
        '-r',
        '--repositories',
        nargs='+',
        default=['mmdet', 'mmcls', 'mmseg', 'mmengine', 'mmcv'],
        type=str,
        help='git repositories name in OpenMMLab')
    parser.add_argument(
        '-b',
        '--branches',
        nargs='+',
        default=['3.x', '1.x', '1.x', 'main', '2.x'],
        type=str,
        help='the branch names of git repositories, the length of branches '
        'must be same as the length of repositories')
    parser.add_argument(
        '-o', '--out', type=str, default='.', help='output path of the file')
    parser.add_argument(
        '--throw-error',
        action='store_true',
        default=False,
        help='whether to throw error when trying to import modules')
    args = parser.parse_args()
    return args


# TODO: Refine
def main():
    args = parse_args()
    repositories = args.repositories
    branches = args.branches
    assert isinstance(repositories, list), \
        'Type of repositories must be list'
    if branches is None:
        branches = [None] * len(repositories)
    assert isinstance(branches, list) and \
           len(branches) == len(repositories), \
           'The length of branches must be same as ' \
           'that of repositories'
    assert isinstance(args.out, str), \
        'The type of output path must be string'
    # save path of file
    mkdir_or_exist(args.out)
    save_path = osp.join(args.out, 'registries_info.md')
    with tempfile.TemporaryDirectory() as tmpdir:
        # multi process init
        pool = Pool(processes=len(repositories))
        multi_proc_input_list = []
        multi_proc_output_list = []
        # get the git repositories
        for branch, repository in zip(branches, repositories):
            repo_name, module_name = parse_repo_name(repository)
            pulldir = osp.join(tmpdir, f'tmp_{repo_name}')
            git_pull_branch(
                repo_name=repo_name, branch_name=branch, pulldir=pulldir)
            multi_proc_input_list.append(
                (repo_name, module_name, branch, pulldir, args.throw_error))
        print('starting the multi process to get the registries')
        for multi_proc_input in multi_proc_input_list:
            multi_proc_output_list.append(
                pool.apply_async(generate_markdown_by_repository,
                                 multi_proc_input))
        pool.close()
        pool.join()
        with open(save_path, 'w', encoding='utf-8') as fw:
            fw.write(f'{markdown_title}\n')
            for multi_proc_output in multi_proc_output_list:
                markdown_str = multi_proc_output.get()
                fw.write(f'{markdown_str}\n')
    print(f'saved registries to the path: {save_path}')


if __name__ == '__main__':
    main()
