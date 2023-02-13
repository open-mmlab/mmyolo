/opt/conda/bin/conda init bash
pip install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io -U openmim
python -m mim install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io "mmengine>=0.3.1"
python -m mim install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io "mmcv>=2.0.0rc1,<2.1.0"
python -m mim install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io "mmdet>=3.0.0rc5,<3.1.0"
python -m mim install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io "mmpose>=1.0.0b0"

python -m pip install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io -r requirements/albu.txt
python -m mim install -i http://devops.io:3141/root/pypi/+simple --trusted-host devops.io -v -e .