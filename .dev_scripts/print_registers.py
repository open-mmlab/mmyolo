# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import os.path as osp
import argparse
import tempfile
import importlib
import pkgutil
import pandas as pd
import numpy as np
from pathlib import Path
from mmengine.utils import mkdir_or_exist, scandir
from mmengine.registry import Registry

# host_addr = 'https://gitee.com/open-mmlab'
host_addr = 'https://github.com/open-mmlab'
tools_list = ['tools']
proxy_names = {
    'mmdet': 'mmdetection',
    'mmseg': 'mmsegmentation',
    'mmcls': 'mmclassification'
}


def parse_repo_name(repo_name):
    proxy_names_rev = dict(zip(proxy_names.values(), proxy_names.keys()))
    repo_name = proxy_names.get(repo_name, repo_name)
    module_name = proxy_names_rev.get(repo_name, repo_name)
    return repo_name, module_name


def git_pull_branch(repo_name, branch_name='', pulldir='.'):
    exec_str = f'cd {pulldir};git init;git pull '
    exec_str += f'{host_addr}/{repo_name}.git'
    if branch_name:
        exec_str += f' {branch_name}'
    returncode = os.system(exec_str)
    if returncode:
        raise RuntimeError(
            f'failed to get the remote repo, code: {returncode}')


def load_modules_from_dir(module_name, module_root, throw_error=False):
    print('loading the modules...')
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
    print('modules were loaded...')
    return module_list, error_dict


def get_registries_from_modules(module_list):
    registries = {}
    objects_set = set()
    # only get the specific registries in module list
    print('getting registries...')
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


def get_pyfiles_from_dir(root):

    def _recurse(_dict, _chain):
        if len(_chain) <= 1:
            _dict[_chain[0]] = None
            return
        _key, *_chain = _chain
        if _key not in _dict:
            _dict[_key] = {}
        _recurse(_dict[_key], _chain)

    # find all scripts in the root directory. (not just ('.py', '.sh'))
    pyfiles = {}
    if osp.isdir(root):
        for pyfile in scandir(root, recursive=True):
            _recurse(pyfiles, Path(pyfile).parts)
    return pyfiles


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
    if (len(_array) % _maxsize_per_group):
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
    html = add_html_title(html, title)
    return html


def tools_to_html(tools_dict, repo_name='', tools_name=''):

    def _recurse(_dict, _connector, _result):
        assert isinstance(_dict, dict), \
            f'unknown recurse type: {_dict} ({type(_dict)})'
        for _k, _v in _dict.items():
            if _v is None:
                if _connector not in _result:
                    _result[_connector] = []
                _result[_connector].append(_k)
            else:
                _recurse(_v, f'{_connector}/{_k}', _result)

    table_data = {}
    title = f'{repo_name.capitalize()} {tools_name.capitalize()}'
    _recurse(tools_dict, tools_name, table_data)
    return registries_to_html(table_data, title)


def dataframe_to_html(dataframe):
    styler = dataframe.style
    styler = styler.hide(axis='index')
    styler = styler.format(na_rep='-')
    styler = styler.set_properties(**{
        'text-align': 'center',
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
    html = f'<div align=\'center\'>{html}</div><br><br>'
    return html


def add_html_title(html, title):
    html = f'<div align=\'center\'><b>{title}</b></div>\n{html}'
    html = f'<details open>{html}</details>'
    return html


def parse_args():
    parser = argparse.ArgumentParser(
        description='print registries in a repository')
    parser.add_argument(
        'repository', type=str,
        help='git repository name in OpenMMLab')
    parser.add_argument(
        '-b', '--branch', type=str,
        help='the branch name of git repository')
    parser.add_argument(
        '-o', '--out', type=str, default='.',
        help='output path of the result')
    parser.add_argument(
        '--throw-error',
        action='store_true',
        default=False,
        help='whether to throw error when trying to import modules')
    parser.add_argument(
        '--without-tools',
        action='store_true',
        default=False,
        help='whether to print the scripts of tools directory')
    args = parser.parse_args()
    return args


# TODO: Refine
def main():
    args = parse_args()
    repo_name, module_name = parse_repo_name(args.repository)
    assert (repo_name not in ['mmengine', 'mmcv']), \
        'mmengine or mmcv is not supported temporarily for querying'
    with tempfile.TemporaryDirectory() as tmpdir:
        # get the registries
        git_pull_branch(
            repo_name=repo_name, branch_name=args.branch, pulldir=tmpdir)
        if tmpdir not in sys.path:
            sys.path.insert(0, tmpdir)
        module_list, error_dict = load_modules_from_dir(
            module_name, tmpdir, throw_error=args.throw_error)
        registries_tree = get_registries_from_modules(module_list)
        if error_dict:
            error_dict_name = 'error_modules'
            assert (error_dict_name not in registries_tree), \
                f'duplicate module name was found: {error_dict_name}'
            registries_tree.update({error_dict_name: error_dict})
        # get the tools files
        if not args.without_tools:
            for tools_name in tools_list:
                assert (tools_name not in registries_tree), \
                    f'duplicate tools name was found: {tools_name}'
                tools_tree = osp.join(tmpdir, tools_name)
                tools_tree = get_pyfiles_from_dir(tools_tree)
                registries_tree.update({tools_name: tools_tree})
        print_tree(registries_tree)
        if args.out:
            # get registries markdown string
            mkdir_or_exist(args.out)
            markdown_str = registries_to_html(
                registries_tree.get(module_name, {}),
                title=f'{repo_name.capitalize()} Module Components')

            # get tools markdown string
            if not args.without_tools:
                for tools_name in tools_list:
                    markdown_str += tools_to_html(
                        registries_tree.get(tools_name, {}),
                        repo_name=repo_name,
                        tools_name=tools_name)

            # save the file
            save_path = osp.join(args.out,
                                 f'registries_info_{module_name.lower()}.md')
            with open(save_path, 'w', encoding='utf-8') as fw:
                fw.write(markdown_str)
            print(f'saved registries to the path: {save_path}')


if __name__ == '__main__':
    main()
