# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import os.path as osp
import argparse
import tempfile
import importlib
import pkgutil
from pathlib import Path
from mmengine.fileio import dump
from mmengine.utils import mkdir_or_exist, scandir
from mmengine.registry import Registry

# host_addr = 'https://github.com/open-mmlab'
# tools_list = ['tools', 'configs']
host_addr = 'https://gitee.com/open-mmlab'
tools_list = ['tools']
proxy_names = {'mmdet': 'mmdetection', 'mmseg': 'mmsegmentation'}


def git_pull_branch(repo_name, branch_name='', pulldir='.'):
    repo_name = proxy_names.get(repo_name, repo_name)
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
    for _root, _dirs, _files in os.walk(module_root):
        if (('__init__.py' not in _files) and
                (osp.split(_root)[1] != '__pycache__')):
            # add __init__.py file to the package
            with open(osp.join(_root, '__init__.py'), 'w') as _:
                pass

    for _finder, _name, _ispkg in pkgutil.walk_packages([module_root],
                                                        prefix=module_name + '.'):
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
            key: str(val) for key, val in _obj.module_dict.items()}
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

    assert osp.exists(root), 'cannot find the recursive dir'
    # find all scripts in the root directory
    pyfiles = {}
    for pyfile in scandir(root, '.py', recursive=True):
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
                sub_tree = _recurse(_val, _connector +
                                    ('   ' if _last else'│  '), n + 1)
            else:
                assert (_val is None), f'unknown print type {_val}'
            tree += '  ' + _connector + \
                ('└─' if _last else '├─') + f'({n}) {_key}' + '\n'
            tree += sub_tree
        return tree

    for _pname, _pdict in print_dict.items():
        print('-' * 100)
        print(f'{_pname}\n' + _recurse(_pdict))


def parse_args():
    parser = argparse.ArgumentParser(
        description='print registries in a repository')
    parser.add_argument(
        'repository', type=str, help='git repository name in OpenMMLab')
    parser.add_argument(
        '-b', '--branch', type=str, help='the branch name of git repository')
    parser.add_argument(
        '--throw-error',
        action='store_true',
        default=False,
        help='whether to throw the import error when trying to import the modules'
    )
    parser.add_argument(
        '--without-tools',
        action='store_true',
        default=False,
        help='whether to print the scripts in tools directory')
    parser.add_argument('--out', type=str, help='output path of result')
    args = parser.parse_args()
    return args


# TODO: Refine
def main():
    args = parse_args()
    repo_name = args.repository
    pwd = osp.split(osp.realpath(__file__))[0]
    with tempfile.TemporaryDirectory(dir=pwd) as tmpdir:
        # get the registries
        git_pull_branch(
            repo_name=repo_name, branch_name=args.branch, pulldir=tmpdir)
        if tmpdir not in sys.path:
            sys.path.insert(0, tmpdir)
        module_list, error_dict = load_modules_from_dir(
            repo_name, tmpdir, throw_error=args.throw_error)
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
                tools_tree = get_pyfiles_from_dir(osp.join(tmpdir, tools_name))
                registries_tree.update({tools_name: tools_tree})
        # print the results
        print_tree(registries_tree)
        # output results
        if args.out:
            mkdir_or_exist(args.out)
            dump(registries_tree, osp.join(args.out, 'registries_info.json'))


if __name__ == '__main__':
    main()
