# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    compare_yolox_mmyolo_loss.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: houbowei <houbowei@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/21 10:29:23 by houbowei          #+#    #+#              #
#    Updated: 2023/02/21 10:48:25 by houbowei         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script is used to compare the loss of yolox and mmyolo.
"""

# Load model
import torch
from yolox.exp import get_exp
exp = get_exp('/home/houbowei/yolox-pose/exps/my_exps/sota_yolox_s_kpt_head_mosaic.py', "yolox-s-kpt-head-mosaic")
yolox_model = exp.get_model()

yolox_loss = yolox_model.head.get_losses

# Load mmyolo model
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmdet.apis import init_detector

register_all_modules()

mmyolo_model = init_detector('./configs/yolox/yoloxkpt_s_8xb8-300e_coco.py')
mmyolo_loss = mmyolo_model.bbox_head.loss_by_feat

# NOTE: yolox and mmyolo use different input format
