# Dummy YOLOv5CSPDarknet Wrapper

This is an example README for community `projects/`. We have provided detailed explanations for each field in the form of html comments, which are visible when you read the source of this README file. If you wish to submit your project to our main repository, then all the fields in this README are mandatory for others to understand what you have achieved in this implementation. For more details, read our [contribution guide](https://mmyolo.readthedocs.io/en/latest/community/contributing.html) or approach us in [Discussions](https://github.com/open-mmlab/mmyolo/discussions).

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This project implements a dummy YOLOv5CSPDarknet wrapper, which literally does nothing new but prints "hello world" during initialization.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMYOLO's root directory, run the following command to train the model:

```bash
python tools/train.py projects/example_project/configs/yolov5_s_dummy-backbone_v61_syncbn_8xb16-300e_coco.py
```

### Testing commands

In MMYOLO's root directory, run the following command to test the model:

```bash
python tools/test.py projects/example_project/configs/yolov5_s_dummy-backbone_v61_syncbn_8xb16-300e_coco.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5#results-and-models)
You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

|                                    Method                                     |       Backbone        | Pretrained Model |  Training set  |   Test set   | #epoch | box AP |         Download         |
| :---------------------------------------------------------------------------: | :-------------------: | :--------------: | :------------: | :----------: | :----: | :----: | :----------------------: |
| [YOLOv5 dummy](configs/yolov5_s_dummy-backbone_v61_syncbn_8xb16-300e_coco.py) | DummyYOLOv5CSPDarknet |        -         | COCO2017 Train | COCO2017 Val |  300   |  37.7  | [model](<>) \| [log](<>) |

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@software{glenn_jocher_2022_7002879,
  author       = {Glenn Jocher and
                  Ayush Chaurasia and
                  Alex Stoken and
                  Jirka Borovec and
                  NanoCode012 and
                  Yonghye Kwon and
                  TaoXie and
                  Kalen Michael and
                  Jiacong Fang and
                  imyhxy and
                  Lorna and
                  Colin Wong and
                  曾逸夫(Zeng Yifu) and
                  Abhiram V and
                  Diego Montes and
                  Zhiqiang Wang and
                  Cristi Fati and
                  Jebastin Nadar and
                  Laughing and
                  UnglvKitDe and
                  tkianai and
                  yxNONG and
                  Piotr Skalski and
                  Adam Hogan and
                  Max Strobel and
                  Mrinal Jain and
                  Lorenzo Mammana and
                  xylieong},
  title        = {{ultralytics/yolov5: v6.2 - YOLOv5 Classification
                   Models, Apple M1, Reproducibility, ClearML and
                   Deci.ai integrations}},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v6.2},
  doi          = {10.5281/zenodo.7002879},
  url          = {https://doi.org/10.5281/zenodo.7002879}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmyolo.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmyolo/blob/27487fd587398348d59eb8c40af740cabee6b7fe/mmyolo/models/layers/yolo_bricks.py#L32-L54) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmyolo/blob/27487fd587398348d59eb8c40af740cabee6b7fe/tests/test_models/test_layers/test_yolo_bricks.py#L13-L34) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inference. [Example](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
