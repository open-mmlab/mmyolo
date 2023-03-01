# 如何给 MMYOLO 贡献代码

欢迎加入 MMYOLO 社区，我们致力于打造最前沿的计算机视觉基础库，我们欢迎任何类型的贡献，包括但不限于

**修复错误**

修复代码实现错误的步骤如下：

1. 如果提交的代码改动较大，建议先提交 issue，并正确描述 issue 的现象、原因和复现方式，讨论后确认修复方案。
2. 修复错误并补充相应的单元测试，提交拉取请求。

**新增功能或组件**

1. 如果新功能或模块涉及较大的代码改动，建议先提交 issue，确认功能的必要性。
2. 实现新增功能并添单元测试，提交拉取请求。

**文档补充**

修复文档可以直接提交拉取请求

添加文档或将文档翻译成其他语言步骤如下

1. 提交 issue，确认添加文档的必要性。
2. 添加文档，提交拉取请求。

## 准备工作

拉取请求工作的命令都是用 Git 去实现的，该章节详细描述 `Git 配置` 以及与 `GitHub 绑定`

### 1. Git 配置

首先，确认电脑是否安装了 Git。Linux 系统和 macOS 系统，一般默认安装 Git，如未安装可在 [Git-Downloads](https://git-scm.com/downloads) 下载。

```shell
# 在命令提示符（cmd）或终端下输入以下命令，查看 Git 版本
git --version
```

其次，检测自己 `Git Config` 是否配置

```shell
# 在命令提示符（cmd）或终端下输入以下命令，查看 Git Config 是否配置
git config --global --list
```

若 `user.name` 和 `user.email` 为空，则输入以下命令进行配置。

```shell
git config --global user.name "这里换上你的用户名"
git config --global user.email "这里换上你的邮箱"
```

最后，在 `git bash` 或者 `终端` 中，输入以下命令生成密钥文件。生成成功后，会在用户目录下出现 `.ssh` 文件，其中 `id_rsa.pub` 是公钥文件。

```shell
# useremail 是 GitHub 的邮箱
ssh-keygen -t rsa -C "useremail"
```

### 2. GitHub 绑定

首先，用记事本打开 `id_rsa.pub` 公钥文件，并复制里面全部内容。

其次，登录 GitHub 账户找到下图位置进行设置。

<img src="https://user-images.githubusercontent.com/90811472/221778382-a075167d-b028-4f68-a1c7-49a8f6f3d97b.png" width="1200">

点击 `New SSH key` 新增一个 SSH keys，将刚才复制的内容粘贴到下图所示的 Key 中，Title 可以写设备名称，最后确认即可。

<img src="https://user-images.githubusercontent.com/90811472/221549754-53670c19-5efe-48b2-9ac5-bafb43891903.png" width="1200">

最后，在 `git bash` 或者 `终端` 中输入以下命令，验证 SSH 是否与 GitHub 账户匹配。如果匹配，输入 `yes` 就成功啦~

```shell
ssh -T git@github.com
```

<img src="https://user-images.githubusercontent.com/90811472/221573637-30e5d04d-955c-4c8f-86ab-ed6608644fc8.png" width="1200">

## 拉取请求工作流

如果你对拉取请求不了解，没关系，接下来的内容将会从零开始，一步一步地指引你如何创建一个拉取请求。如果你想深入了解拉取请求的开发模式，可以参考 github [官方文档](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

### 1. 复刻仓库

当你第一次提交拉取请求时，先复刻 OpenMMLab 原代码库，点击 GitHub 页面右上角的 **Fork** 按钮，复刻后的代码库将会出现在你的 GitHub 个人主页下。

<img src="https://user-images.githubusercontent.com/27466624/204301143-2d262d2c-28b3-4060-8576-21d9f4237f2f.png" width="1200">

将代码克隆到本地

```shell
git clone git@github.com:{username}/mmyolo.git
```

进入项目并添加原代码库为上游代码库

```bash
cd mmyolo
git remote add upstream git@github.com:open-mmlab/mmyolo
```

检查 remote 是否添加成功，在终端输入 `git remote -v`

```bash
origin	git@github.com:{username}/mmyolo.git (fetch)
origin	git@github.com:{username}/mmyolo.git (push)
upstream	git@github.com:open-mmlab/mmyolo (fetch)
upstream	git@github.com:open-mmlab/mmyolo (push)
```

```{note}
这里对 origin 和 upstream 进行一个简单的介绍，当我们使用 git clone 来克隆代码时，会默认创建一个 origin 的 remote，它指向我们克隆的代码库地址，而 upstream 则是我们自己添加的，用来指向原始代码库地址。当然如果你不喜欢他叫 upstream，也可以自己修改，比如叫 open-mmlab。我们通常向 origin 提交代码（即 fork 下来的远程仓库），然后向 upstream 提交一个 pull request。如果提交的代码和最新的代码发生冲突，再从 upstream 拉取最新的代码，和本地分支解决冲突，再提交到 origin。
```

### 2. 配置 pre-commit

在本地开发环境中，我们使用 [pre-commit](https://pre-commit.com/#intro) 来检查代码风格，以确保代码风格的统一。在提交代码，需要先安装 pre-commit（需要在 MMYOLO 目录下执行）:

```shell
pip install -U pre-commit
pre-commit install
```

检查 pre-commit 是否配置成功，并安装 `.pre-commit-config.yaml` 中的钩子：

```shell
pre-commit run --all-files
```

<img src="https://user-images.githubusercontent.com/57566630/173660750-3df20a63-cb66-4d33-a986-1f643f1d8aaf.png" width="1200">

<img src="https://user-images.githubusercontent.com/57566630/202368856-0465a90d-8fce-4345-918e-67b8b9c82614.png" width="1200">

```{note}
如果你是中国用户，由于网络原因，可能会出现安装失败的情况，这时可以使用国内源

pre-commit install -c .pre-commit-config-zh-cn.yaml

pre-commit run --all-files -c .pre-commit-config-zh-cn.yaml
```

如果安装过程被中断，可以重复执行 `pre-commit run ...` 继续安装。

如果提交的代码不符合代码风格规范，pre-commit 会发出警告，并自动修复部分错误。

<img src="https://user-images.githubusercontent.com/57566630/202369176-67642454-0025-4023-a095-263529107aa3.png" width="1200">

如果我们想临时绕开 pre-commit 的检查提交一次代码，可以在 `git commit` 时加上 `--no-verify`（需要保证最后推送至远程仓库的代码能够通过 pre-commit 检查）。

```shell
git commit -m "xxx" --no-verify
```

### 3. 创建开发分支

安装完 pre-commit 之后，我们需要基于 dev 创建开发分支，建议的分支命名规则为 `username/pr_name`。

```shell
git checkout -b yhc/refactor_contributing_doc
```

在后续的开发中，如果本地仓库的 dev 分支落后于 upstream 的 dev 分支，我们需要先拉取 upstream 的代码进行同步，再执行上面的命令

```shell
git pull upstream dev
```

### 4. 提交代码并在本地通过单元测试

- MMYOLO 引入了 mypy 来做静态类型检查，以增加代码的鲁棒性。因此我们在提交代码时，需要补充 Type Hints。具体规则可以参考[教程](https://zhuanlan.zhihu.com/p/519335398)。

- 提交的代码同样需要通过单元测试

  ```shell
  # 通过全量单元测试
  pytest tests

  # 我们需要保证提交的代码能够通过修改模块的单元测试，以 yolov5_coco dataset 为例
  pytest tests/test_datasets/test_yolov5_coco.py
  ```

  如果你由于缺少依赖无法运行修改模块的单元测试，可以参考[指引-单元测试](#单元测试)

- 如果修改/添加了文档，参考[指引](#文档渲染)确认文档渲染正常。

### 5. 推送代码到远程

代码通过单元测试和 pre-commit 检查后，将代码推送到远程仓库，如果是第一次推送，可以在 `git push` 后加上 `-u` 参数以关联远程分支

```shell
git push -u origin {branch_name}
```

这样下次就可以直接使用 `git push` 命令推送代码了，而无需指定分支和远程仓库。

### 6. 提交拉取请求（PR）

(1) 在 GitHub 的 Pull request 界面创建拉取请求
<img src="https://user-images.githubusercontent.com/27466624/204302289-d1e54901-8f27-4934-923f-fda800ff9851.png" width="1200">

(2) 根据指引修改 PR 描述，以便于其他开发者更好地理解你的修改

```{note}
注意在 PR branch 左侧的 base 需要修改为 dev 分支
```

<img src="https://user-images.githubusercontent.com/62822224/216594960-a2292b9d-2b7c-4861-b4c5-362a9458b194.png" width="1200">

描述规范详见[拉取请求规范](#拉取请求规范)

&#160;

**注意事项**

(a) PR 描述应该包含修改理由、修改内容以及修改后带来的影响，并关联相关 Issue（具体方式见[文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)）

(b) 如果是第一次为 OpenMMLab 做贡献，需要签署 CLA

<img src="https://user-images.githubusercontent.com/57566630/167307569-a794b967-6e28-4eac-a942-00deb657815f.png" width="1200">

(c) 检查提交的 PR 是否通过 CI（集成测试）

<img src="https://user-images.githubusercontent.com/27466624/204303753-900de590-ddd1-4be2-8e43-8dc09f127f5d.png" width="1200">

MMYOLO 会在 Linux 上，基于不同版本的 Python、PyTorch 对提交的代码进行单元测试，以保证代码的正确性，如果有任何一个没有通过，我们可点击上图中的 `Details` 来查看具体的测试信息，以便于我们修改代码。

(3) 如果 PR 通过了 CI，那么就可以等待其他开发者的 review，并根据 reviewer 的意见，修改代码，并重复 [4](#4-提交代码并本地通过单元测试)-[5](#5-推送代码到远程) 步骤，直到 reviewer 同意合入 PR。

<img src="https://user-images.githubusercontent.com/57566630/202145400-cc2cd8c4-10b0-472f-ba37-07e6f50acc67.png" width="1200">

所有 reviewer 同意合入 PR 后，我们会尽快将 PR 合并到 dev 分支。

### 7. 解决冲突

随着时间的推移，我们的代码库会不断更新，这时候，如果你的 PR 与 dev 分支存在冲突，你需要解决冲突，解决冲突的方式有两种：

```shell
git fetch --all --prune
git rebase upstream/dev
```

或者

```shell
git fetch --all --prune
git merge upstream/dev
```

如果你非常善于处理冲突，那么可以使用 rebase 的方式来解决冲突，因为这能够保证你的 commit log 的整洁。如果你不太熟悉 `rebase` 的使用，那么可以使用 `merge` 的方式来解决冲突。

## 指引

### 单元测试

在提交修复代码错误或新增特性的拉取请求时，我们应该尽可能的让单元测试覆盖所有提交的代码，计算单元测试覆盖率的方法如下

```shell
python -m coverage run -m pytest /path/to/test_file
python -m coverage html
# check file in htmlcov/index.html
```

### 文档渲染

在提交修复代码错误或新增特性的拉取请求时，可能会需要修改/新增模块的 docstring。我们需要确认渲染后的文档样式是正确的。
本地生成渲染后的文档的方法如下

```shell
pip install -r requirements/docs.txt
cd docs/zh_cn/
# or docs/en
make html
# check file in ./docs/zh_cn/_build/html/index.html
```

## 代码风格

### Python

[PEP8](https://www.python.org/dev/peps/pep-0008/) 作为 OpenMMLab 算法库首选的代码规范，我们使用以下工具检查和格式化代码

- [flake8](https://github.com/PyCQA/flake8)：Python 官方发布的代码规范检查工具，是多个检查工具的封装
- [isort](https://github.com/timothycrosley/isort)：自动调整模块导入顺序的工具
- [yapf](https://github.com/google/yapf)：Google 发布的代码规范检查工具
- [codespell](https://github.com/codespell-project/codespell)：检查单词拼写是否有误
- [mdformat](https://github.com/executablebooks/mdformat)：检查 markdown 文件的工具
- [docformatter](https://github.com/myint/docformatter)：格式化 docstring 的工具

yapf 和 isort 的配置可以在 [setup.cfg](../../../setup.cfg) 找到

通过配置 [pre-commit hook](https://pre-commit.com/) ，我们可以在提交代码时自动检查和格式化 `flake8`、`yapf`、`isort`、`trailing whitespaces`、`markdown files`，
修复 `end-of-files`、`double-quoted-strings`、`python-encoding-pragma`、`mixed-line-ending`，调整 `requirments.txt` 的包顺序。
pre-commit 钩子的配置可以在 [.pre-commit-config](../../../.pre-commit-config.yaml) 找到。

pre-commit 具体的安装使用方式见[拉取请求](#2-配置-pre-commit)。

更具体的规范请参考 [OpenMMLab 代码规范](../notes/code_style.md)。

### C++ and CUDA

C++ 和 CUDA 的代码规范遵从 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

## 拉取请求规范

1. 使用 [pre-commit hook](https://pre-commit.com)，尽量减少代码风格相关问题

2. 一个`拉取请求`对应一个短期分支

3. 粒度要细，一个`拉取请求`只做一件事情，避免超大的`拉取请求`

   - Bad：实现 Faster R-CNN
   - Acceptable：给 Faster R-CNN 添加一个 box head
   - Good：给 box head 增加一个参数来支持自定义的 conv 层数

4. 每次 Commit 时需要提供清晰且有意义 commit 信息

5. 提供清晰且有意义的`拉取请求`描述

   - 标题写明白任务名称，一般格式:\[Prefix\] Short description of the pull request (Suffix)
   - prefix：新增功能 \[Feature\], 修 bug \[Fix\], 文档相关 \[Docs\], 开发中 \[WIP\] (暂时不会被 review)
   - 描述里介绍`拉取请求`的主要修改内容，结果，以及对其他部分的影响, 参考`拉取请求`模板
   - 关联相关的`议题` (issue) 和其他`拉取请求`

6. 如果引入了其他三方库，或借鉴了三方库的代码，请确认他们的许可证和 mmyolo 兼容，并在借鉴的代码上补充 `This code is inspired from http://`
