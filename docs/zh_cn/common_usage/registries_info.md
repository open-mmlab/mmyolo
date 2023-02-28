# MM 系列开源库注册表

（注意：本文档是通过 .dev_scripts/print_registers.py 脚本自动生成）

## MMdetection (3.0.0rc6)

<details open><div align='center'><b>MMdetection Module Components</b></div>
<div align='center'>
<style type="text/css">
#T_2775b thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_2775b_row0_col0, #T_2775b_row0_col1, #T_2775b_row0_col2, #T_2775b_row0_col3, #T_2775b_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_2775b">
  <thead>
    <tr>
      <th id="T_2775b_level0_col0" class="col_heading level0 col0" >visualizer</th>
      <th id="T_2775b_level0_col1" class="col_heading level0 col1" >optimizer constructor</th>
      <th id="T_2775b_level0_col2" class="col_heading level0 col2" >loop</th>
      <th id="T_2775b_level0_col3" class="col_heading level0 col3" >parameter scheduler</th>
      <th id="T_2775b_level0_col4" class="col_heading level0 col4" >data sampler</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_2775b_row0_col0" class="data row0 col0" ><ul><li>DetLocalVisualizer</li></ul></td>
      <td id="T_2775b_row0_col1" class="data row0 col1" ><ul><li>LearningRateDecayOptimizerConstructor</li></ul></td>
      <td id="T_2775b_row0_col2" class="data row0 col2" ><ul><li>TeacherStudentValLoop</li></ul></td>
      <td id="T_2775b_row0_col3" class="data row0 col3" ><ul><li>QuadraticWarmupParamScheduler</li><li>QuadraticWarmupLR</li><li>QuadraticWarmupMomentum</li></ul></td>
      <td id="T_2775b_row0_col4" class="data row0 col4" ><ul><li>AspectRatioBatchSampler</li><li>ClassAwareSampler</li><li>MultiSourceSampler</li><li>GroupMultiSourceSampler</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_00a84 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_00a84_row0_col0, #T_00a84_row0_col1, #T_00a84_row0_col2, #T_00a84_row0_col3, #T_00a84_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_00a84">
  <thead>
    <tr>
      <th id="T_00a84_level0_col0" class="col_heading level0 col0" >metric</th>
      <th id="T_00a84_level0_col1" class="col_heading level0 col1" >hook</th>
      <th id="T_00a84_level0_col2" class="col_heading level0 col2" >dataset</th>
      <th id="T_00a84_level0_col3" class="col_heading level0 col3" >task util (part 1)</th>
      <th id="T_00a84_level0_col4" class="col_heading level0 col4" >task util (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_00a84_row0_col0" class="data row0 col0" ><ul><li>CityScapesMetric</li><li>CocoMetric</li><li>CocoOccludedSeparatedMetric</li><li>CocoPanopticMetric</li><li>CrowdHumanMetric</li><li>DumpDetResults</li><li>DumpProposals</li><li>LVISMetric</li><li>OpenImagesMetric</li><li>VOCMetric</li></ul></td>
      <td id="T_00a84_row0_col1" class="data row0 col1" ><ul><li>CheckInvalidLossHook</li><li>MeanTeacherHook</li><li>MemoryProfilerHook</li><li>NumClassCheckHook</li><li>PipelineSwitchHook</li><li>SetEpochInfoHook</li><li>SyncNormHook</li><li>DetVisualizationHook</li><li>YOLOXModeSwitchHook</li><li>FastStopTrainingHook</li></ul></td>
      <td id="T_00a84_row0_col2" class="data row0 col2" ><ul><li>BaseDetDataset</li><li>CocoDataset</li><li>CityscapesDataset</li><li>CocoPanopticDataset</li><li>CrowdHumanDataset</li><li>MultiImageMixDataset</li><li>DeepFashionDataset</li><li>LVISV05Dataset</li><li>LVISDataset</li><li>LVISV1Dataset</li><li>Objects365V1Dataset</li><li>Objects365V2Dataset</li><li>OpenImagesDataset</li><li>OpenImagesChallengeDataset</li><li>XMLDataset</li><li>VOCDataset</li><li>WIDERFaceDataset</li></ul></td>
      <td id="T_00a84_row0_col3" class="data row0 col3" ><ul><li>MaxIoUAssigner</li><li>ApproxMaxIoUAssigner</li><li>ATSSAssigner</li><li>CenterRegionAssigner</li><li>DynamicSoftLabelAssigner</li><li>GridAssigner</li><li>HungarianAssigner</li><li>BboxOverlaps2D</li><li>BBoxL1Cost</li><li>IoUCost</li><li>ClassificationCost</li><li>FocalLossCost</li><li>DiceCost</li><li>CrossEntropyLossCost</li><li>MultiInstanceAssigner</li></ul></td>
      <td id="T_00a84_row0_col4" class="data row0 col4" ><ul><li>PointAssigner</li><li>AnchorGenerator</li><li>SSDAnchorGenerator</li><li>LegacyAnchorGenerator</li><li>LegacySSDAnchorGenerator</li><li>YOLOAnchorGenerator</li><li>PointGenerator</li><li>MlvlPointGenerator</li><li>RegionAssigner</li><li>SimOTAAssigner</li><li>TaskAlignedAssigner</li><li>UniformAssigner</li><li>BucketingBBoxCoder</li><li>DeltaXYWHBBoxCoder</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_cd536 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_cd536_row0_col0, #T_cd536_row0_col1, #T_cd536_row0_col2, #T_cd536_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_cd536">
  <thead>
    <tr>
      <th id="T_cd536_level0_col0" class="col_heading level0 col0" >task util (part 3)</th>
      <th id="T_cd536_level0_col1" class="col_heading level0 col1" >transform (part 1)</th>
      <th id="T_cd536_level0_col2" class="col_heading level0 col2" >transform (part 2)</th>
      <th id="T_cd536_level0_col3" class="col_heading level0 col3" >transform (part 3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_cd536_row0_col0" class="data row0 col0" ><ul><li>DistancePointBBoxCoder</li><li>LegacyDeltaXYWHBBoxCoder</li><li>PseudoBBoxCoder</li><li>TBLRBBoxCoder</li><li>YOLOBBoxCoder</li><li>CombinedSampler</li><li>RandomSampler</li><li>InstanceBalancedPosSampler</li><li>IoUBalancedNegSampler</li><li>MaskPseudoSampler</li><li>MultiInsRandomSampler</li><li>OHEMSampler</li><li>PseudoSampler</li><li>ScoreHLRSampler</li></ul></td>
      <td id="T_cd536_row0_col1" class="data row0 col1" ><ul><li>AutoAugment</li><li>RandAugment</li><li>ColorTransform</li><li>Color</li><li>Brightness</li><li>Contrast</li><li>Sharpness</li><li>Solarize</li><li>SolarizeAdd</li><li>Posterize</li><li>Equalize</li><li>AutoContrast</li><li>Invert</li><li>PackDetInputs</li><li>ToTensor</li><li>ImageToTensor</li><li>Transpose</li><li>WrapFieldsToLists</li><li>GeomTransform</li><li>ShearX</li></ul></td>
      <td id="T_cd536_row0_col2" class="data row0 col2" ><ul><li>ShearY</li><li>Rotate</li><li>TranslateX</li><li>TranslateY</li><li>InstaBoost</li><li>LoadImageFromNDArray</li><li>LoadMultiChannelImageFromFiles</li><li>LoadAnnotations</li><li>LoadPanopticAnnotations</li><li>LoadProposals</li><li>FilterAnnotations</li><li>LoadEmptyAnnotations</li><li>InferencerLoader</li><li>Resize</li><li>FixShapeResize</li><li>RandomFlip</li><li>RandomShift</li><li>Pad</li><li>RandomCrop</li></ul></td>
      <td id="T_cd536_row0_col3" class="data row0 col3" ><ul><li>SegRescale</li><li>PhotoMetricDistortion</li><li>Expand</li><li>MinIoURandomCrop</li><li>Corrupt</li><li>Albu</li><li>RandomCenterCropPad</li><li>CutOut</li><li>Mosaic</li><li>MixUp</li><li>RandomAffine</li><li>YOLOXHSVRandomAug</li><li>CopyPaste</li><li>RandomErasing</li><li>CachedMosaic</li><li>CachedMixUp</li><li>MultiBranch</li><li>RandomOrder</li><li>ProposalBroadcaster</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_bb293 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_bb293_row0_col0, #T_bb293_row0_col1, #T_bb293_row0_col2, #T_bb293_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_bb293">
  <thead>
    <tr>
      <th id="T_bb293_level0_col0" class="col_heading level0 col0" >model (part 1)</th>
      <th id="T_bb293_level0_col1" class="col_heading level0 col1" >model (part 2)</th>
      <th id="T_bb293_level0_col2" class="col_heading level0 col2" >model (part 3)</th>
      <th id="T_bb293_level0_col3" class="col_heading level0 col3" >model (part 4)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_bb293_row0_col0" class="data row0 col0" ><ul><li>SiLU</li><li>DropBlock</li><li>ExpMomentumEMA</li><li>SinePositionalEncoding</li><li>LearnedPositionalEncoding</li><li>DynamicConv</li><li>MSDeformAttnPixelDecoder</li><li>Linear</li><li>NormedLinear</li><li>NormedConv2d</li><li>PixelDecoder</li><li>TransformerEncoderPixelDecoder</li><li>CSPDarknet</li><li>CSPNeXt</li><li>Darknet</li><li>ResNet</li><li>ResNetV1d</li><li>DetectoRS_ResNet</li><li>DetectoRS_ResNeXt</li><li>EfficientNet</li></ul></td>
      <td id="T_bb293_row0_col1" class="data row0 col1" ><ul><li>HourglassNet</li><li>HRNet</li><li>MobileNetV2</li><li>PyramidVisionTransformer</li><li>PyramidVisionTransformerV2</li><li>ResNeXt</li><li>RegNet</li><li>Res2Net</li><li>ResNeSt</li><li>BFP</li><li>ChannelMapper</li><li>CSPNeXtPAFPN</li><li>CTResNetNeck</li><li>DilatedEncoder</li><li>DyHead</li><li>FPG</li><li>FPN</li><li>FPN_CARAFE</li><li>HRFPN</li><li>NASFPN</li></ul></td>
      <td id="T_bb293_row0_col2" class="data row0 col2" ><ul><li>NASFCOS_FPN</li><li>PAFPN</li><li>RFP</li><li>SSDNeck</li><li>SSH</li><li>YOLOV3Neck</li><li>YOLOXPAFPN</li><li>SSDVGG</li><li>SwinTransformer</li><li>TridentResNet</li><li>DetDataPreprocessor</li><li>BatchSyncRandomResize</li><li>BatchFixedSizePad</li><li>MultiBranchDataPreprocessor</li><li>BatchResize</li><li>BoxInstDataPreprocessor</li><li>AnchorFreeHead</li><li>AnchorHead</li><li>ATSSHead</li><li>FCOSHead</li></ul></td>
      <td id="T_bb293_row0_col3" class="data row0 col3" ><ul><li>AutoAssignHead</li><li>CondInstBboxHead</li><li>CondInstMaskHead</li><li>BoxInstBboxHead</li><li>BoxInstMaskHead</li><li>RPNHead</li><li>StageCascadeRPNHead</li><li>CascadeRPNHead</li><li>CenterNetHead</li><li>CenterNetUpdateHead</li><li>CornerHead</li><li>CentripetalHead</li><li>DETRHead</li><li>ConditionalDETRHead</li><li>DABDETRHead</li><li>DDODHead</li><li>DeformableDETRHead</li><li>DINOHead</li><li>EmbeddingRPNHead</li><li>FoveaHead</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_39d04 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_39d04_row0_col0, #T_39d04_row0_col1, #T_39d04_row0_col2, #T_39d04_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_39d04">
  <thead>
    <tr>
      <th id="T_39d04_level0_col0" class="col_heading level0 col0" >model (part 5)</th>
      <th id="T_39d04_level0_col1" class="col_heading level0 col1" >model (part 6)</th>
      <th id="T_39d04_level0_col2" class="col_heading level0 col2" >model (part 7)</th>
      <th id="T_39d04_level0_col3" class="col_heading level0 col3" >model (part 8)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_39d04_row0_col0" class="data row0 col0" ><ul><li>RetinaHead</li><li>FreeAnchorRetinaHead</li><li>AssociativeEmbeddingLoss</li><li>BalancedL1Loss</li><li>CrossEntropyLoss</li><li>DiceLoss</li><li>FocalLoss</li><li>GaussianFocalLoss</li><li>QualityFocalLoss</li><li>DistributionFocalLoss</li><li>GHMC</li><li>GHMR</li><li>IoULoss</li><li>BoundedIoULoss</li><li>GIoULoss</li><li>DIoULoss</li><li>CIoULoss</li><li>EIoULoss</li><li>KnowledgeDistillationKLDivLoss</li><li>MSELoss</li></ul></td>
      <td id="T_39d04_row0_col1" class="data row0 col1" ><ul><li>SeesawLoss</li><li>SmoothL1Loss</li><li>L1Loss</li><li>VarifocalLoss</li><li>FSAFHead</li><li>GuidedAnchorHead</li><li>GARetinaHead</li><li>GARPNHead</li><li>GFLHead</li><li>PAAHead</li><li>LADHead</li><li>LDHead</li><li>MaskFormerHead</li><li>Mask2FormerHead</li><li>NASFCOSHead</li><li>PISARetinaHead</li><li>SSDHead</li><li>PISASSDHead</li><li>RepPointsHead</li><li>RetinaSepBNHead</li></ul></td>
      <td id="T_39d04_row0_col2" class="data row0 col2" ><ul><li>RTMDetHead</li><li>RTMDetSepBNHead</li><li>RTMDetInsHead</li><li>RTMDetInsSepBNHead</li><li>SABLRetinaHead</li><li>SOLOHead</li><li>DecoupledSOLOHead</li><li>DecoupledSOLOLightHead</li><li>SOLOV2Head</li><li>TOODHead</li><li>VFNetHead</li><li>YOLACTHead</li><li>YOLACTProtonet</li><li>YOLOV3Head</li><li>YOLOFHead</li><li>YOLOXHead</li><li>SingleStageDetector</li><li>ATSS</li><li>AutoAssign</li></ul></td>
      <td id="T_39d04_row0_col3" class="data row0 col3" ><ul><li>DetectionTransformer</li><li>SingleStageInstanceSegmentor</li><li>BoxInst</li><li>TwoStageDetector</li><li>CascadeRCNN</li><li>CenterNet</li><li>CondInst</li><li>DETR</li><li>ConditionalDETR</li><li>CornerNet</li><li>CrowdDet</li><li>Detectron2Wrapper</li><li>DABDETR</li><li>DDOD</li><li>DeformableDETR</li><li>DINO</li><li>FastRCNN</li><li>FasterRCNN</li><li>FCOS</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_1d4b5 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_1d4b5_row0_col0, #T_1d4b5_row0_col1, #T_1d4b5_row0_col2, #T_1d4b5_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_1d4b5">
  <thead>
    <tr>
      <th id="T_1d4b5_level0_col0" class="col_heading level0 col0" >model (part 9)</th>
      <th id="T_1d4b5_level0_col1" class="col_heading level0 col1" >model (part 10)</th>
      <th id="T_1d4b5_level0_col2" class="col_heading level0 col2" >model (part 11)</th>
      <th id="T_1d4b5_level0_col3" class="col_heading level0 col3" >model (part 12)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_1d4b5_row0_col0" class="data row0 col0" ><ul><li>FOVEA</li><li>FSAF</li><li>GFL</li><li>GridRCNN</li><li>HybridTaskCascade</li><li>KnowledgeDistillationSingleStageDetector</li><li>LAD</li><li>MaskFormer</li><li>Mask2Former</li><li>MaskRCNN</li><li>MaskScoringRCNN</li><li>NASFCOS</li><li>PAA</li><li>TwoStagePanopticSegmentor</li><li>PanopticFPN</li><li>PointRend</li><li>SparseRCNN</li><li>QueryInst</li><li>RepPointsDetector</li></ul></td>
      <td id="T_1d4b5_row0_col1" class="data row0 col1" ><ul><li>RetinaNet</li><li>RPN</li><li>RTMDet</li><li>SCNet</li><li>SemiBaseDetector</li><li>SoftTeacher</li><li>SOLO</li><li>SOLOv2</li><li>TOOD</li><li>TridentFasterRCNN</li><li>VFNet</li><li>YOLACT</li><li>YOLOV3</li><li>YOLOF</li><li>YOLOX</li><li>BBoxHead</li><li>ConvFCBBoxHead</li><li>Shared2FCBBoxHead</li><li>Shared4Conv1FCBBoxHead</li></ul></td>
      <td id="T_1d4b5_row0_col2" class="data row0 col2" ><ul><li>DIIHead</li><li>DoubleConvFCBBoxHead</li><li>MultiInstanceBBoxHead</li><li>SABLHead</li><li>SCNetBBoxHead</li><li>CascadeRoIHead</li><li>StandardRoIHead</li><li>DoubleHeadRoIHead</li><li>DynamicRoIHead</li><li>GridRoIHead</li><li>HybridTaskCascadeRoIHead</li><li>FCNMaskHead</li><li>CoarseMaskHead</li><li>DynamicMaskHead</li><li>FeatureRelayHead</li><li>FusedSemanticHead</li><li>GlobalContextHead</li><li>GridHead</li><li>HTCMaskHead</li></ul></td>
      <td id="T_1d4b5_row0_col3" class="data row0 col3" ><ul><li>MaskPointHead</li><li>MaskIoUHead</li><li>SCNetMaskHead</li><li>SCNetSemanticHead</li><li>MaskScoringRoIHead</li><li>MultiInstanceRoIHead</li><li>PISARoIHead</li><li>PointRendRoIHead</li><li>GenericRoIExtractor</li><li>SingleRoIExtractor</li><li>SCNetRoIHead</li><li>ResLayer</li><li>SparseRoIHead</li><li>TridentRoIHead</li><li>BaseSemanticHead</li><li>PanopticFPNHead</li><li>BasePanopticFusionHead</li><li>HeuristicFusionHead</li><li>MaskFormerFusionHead</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>
<details open><div align='center'><b>MMdetection Tools</b></div>
<div align='center'>
<style type="text/css">
#T_a7e06 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_a7e06_row0_col0, #T_a7e06_row0_col1, #T_a7e06_row0_col2, #T_a7e06_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_a7e06">
  <thead>
    <tr>
      <th id="T_a7e06_level0_col0" class="col_heading level0 col0" >tools/dataset_converters</th>
      <th id="T_a7e06_level0_col1" class="col_heading level0 col1" >tools/deployment</th>
      <th id="T_a7e06_level0_col2" class="col_heading level0 col2" >tools</th>
      <th id="T_a7e06_level0_col3" class="col_heading level0 col3" >tools/misc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_a7e06_row0_col0" class="data row0 col0" ><ul><li>pascal_voc.py</li><li>images2coco.py</li><li>cityscapes.py</li></ul></td>
      <td id="T_a7e06_row0_col1" class="data row0 col1" ><ul><li>mmdet2torchserve.py</li><li>test_torchserver.py</li><li>mmdet_handler.py</li></ul></td>
      <td id="T_a7e06_row0_col2" class="data row0 col2" ><ul><li>dist_test.sh</li><li>slurm_test.sh</li><li>test.py</li><li>dist_train.sh</li><li>train.py</li><li>slurm_train.sh</li></ul></td>
      <td id="T_a7e06_row0_col3" class="data row0 col3" ><ul><li>download_dataset.py</li><li>get_image_metas.py</li><li>gen_coco_panoptic_test_info.py</li><li>split_coco.py</li><li>get_crowdhuman_id_hw.py</li><li>print_config.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_63a8e thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_63a8e_row0_col0, #T_63a8e_row0_col1, #T_63a8e_row0_col2, #T_63a8e_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_63a8e">
  <thead>
    <tr>
      <th id="T_63a8e_level0_col0" class="col_heading level0 col0" >tools/model_converters</th>
      <th id="T_63a8e_level0_col1" class="col_heading level0 col1" >tools/analysis_tools</th>
      <th id="T_63a8e_level0_col2" class="col_heading level0 col2" >.dev_scripts (part 1)</th>
      <th id="T_63a8e_level0_col3" class="col_heading level0 col3" >.dev_scripts (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_63a8e_row0_col0" class="data row0 col0" ><ul><li>upgrade_model_version.py</li><li>upgrade_ssd_version.py</li><li>detectron2_to_mmdet.py</li><li>selfsup2mmdet.py</li><li>detectron2pytorch.py</li><li>regnet2mmdet.py</li><li>publish_model.py</li></ul></td>
      <td id="T_63a8e_row0_col1" class="data row0 col1" ><ul><li>benchmark.py</li><li>eval_metric.py</li><li>robustness_eval.py</li><li>confusion_matrix.py</li><li>optimize_anchors.py</li><li>browse_dataset.py</li><li>test_robustness.py</li><li>coco_error_analysis.py</li><li>coco_occluded_separated_recall.py</li><li>analyze_results.py</li><li>analyze_logs.py</li><li>get_flops.py</li></ul></td>
      <td id="T_63a8e_row0_col2" class="data row0 col2" ><ul><li>convert_test_benchmark_script.py</li><li>gather_test_benchmark_metric.py</li><li>benchmark_valid_flops.py</li><li>benchmark_train.py</li><li>test_benchmark.sh</li><li>download_checkpoints.py</li><li>benchmark_test_image.py</li><li>covignore.cfg</li><li>benchmark_full_models.txt</li><li>test_init_backbone.py</li><li>batch_train_list.txt</li><li>diff_coverage_test.sh</li></ul></td>
      <td id="T_63a8e_row0_col3" class="data row0 col3" ><ul><li>batch_test_list.py</li><li>linter.sh</li><li>gather_train_benchmark_metric.py</li><li>train_benchmark.sh</li><li>benchmark_inference_fps.py</li><li>benchmark_options.py</li><li>check_links.py</li><li>benchmark_test.py</li><li>benchmark_train_models.txt</li><li>convert_train_benchmark_script.py</li><li>gather_models.py</li><li>benchmark_filter.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>

## MMclassification (1.0.0rc5)

<details open><div align='center'><b>MMclassification Module Components</b></div>
<div align='center'>
<style type="text/css">
#T_0c9eb thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_0c9eb_row0_col0, #T_0c9eb_row0_col1, #T_0c9eb_row0_col2, #T_0c9eb_row0_col3, #T_0c9eb_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_0c9eb">
  <thead>
    <tr>
      <th id="T_0c9eb_level0_col0" class="col_heading level0 col0" >visualizer</th>
      <th id="T_0c9eb_level0_col1" class="col_heading level0 col1" >data sampler</th>
      <th id="T_0c9eb_level0_col2" class="col_heading level0 col2" >optimizer</th>
      <th id="T_0c9eb_level0_col3" class="col_heading level0 col3" >batch augment</th>
      <th id="T_0c9eb_level0_col4" class="col_heading level0 col4" >metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_0c9eb_row0_col0" class="data row0 col0" ><ul><li>ClsVisualizer</li></ul></td>
      <td id="T_0c9eb_row0_col1" class="data row0 col1" ><ul><li>RepeatAugSampler</li></ul></td>
      <td id="T_0c9eb_row0_col2" class="data row0 col2" ><ul><li>Adan</li><li>Lamb</li></ul></td>
      <td id="T_0c9eb_row0_col3" class="data row0 col3" ><ul><li>Mixup</li><li>CutMix</li><li>ResizeMix</li></ul></td>
      <td id="T_0c9eb_row0_col4" class="data row0 col4" ><ul><li>Accuracy</li><li>SingleLabelMetric</li><li>MultiLabelMetric</li><li>AveragePrecision</li><li>MultiTasksMetric</li><li>VOCMultiLabelMetric</li><li>VOCAveragePrecision</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_896f9 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_896f9_row0_col0, #T_896f9_row0_col1, #T_896f9_row0_col2, #T_896f9_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_896f9">
  <thead>
    <tr>
      <th id="T_896f9_level0_col0" class="col_heading level0 col0" >hook</th>
      <th id="T_896f9_level0_col1" class="col_heading level0 col1" >dataset</th>
      <th id="T_896f9_level0_col2" class="col_heading level0 col2" >transform (part 1)</th>
      <th id="T_896f9_level0_col3" class="col_heading level0 col3" >transform (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_896f9_row0_col0" class="data row0 col0" ><ul><li>ClassNumCheckHook</li><li>EMAHook</li><li>SetAdaptiveMarginsHook</li><li>PreciseBNHook</li><li>PrepareProtoBeforeValLoopHook</li><li>SwitchRecipeHook</li><li>VisualizationHook</li></ul></td>
      <td id="T_896f9_row0_col1" class="data row0 col1" ><ul><li>BaseDataset</li><li>CIFAR10</li><li>CIFAR100</li><li>CUB</li><li>CustomDataset</li><li>KFoldDataset</li><li>ImageNet</li><li>ImageNet21k</li><li>MNIST</li><li>FashionMNIST</li><li>MultiLabelDataset</li><li>MultiTaskDataset</li><li>VOC</li></ul></td>
      <td id="T_896f9_row0_col2" class="data row0 col2" ><ul><li>AutoAugment</li><li>RandAugment</li><li>Shear</li><li>Translate</li><li>Rotate</li><li>AutoContrast</li><li>Invert</li><li>Equalize</li><li>Solarize</li><li>SolarizeAdd</li><li>Posterize</li><li>Contrast</li><li>ColorTransform</li><li>Brightness</li><li>Sharpness</li><li>Cutout</li></ul></td>
      <td id="T_896f9_row0_col3" class="data row0 col3" ><ul><li>PackClsInputs</li><li>PackMultiTaskInputs</li><li>Transpose</li><li>ToPIL</li><li>ToNumpy</li><li>Collect</li><li>RandomCrop</li><li>RandomResizedCrop</li><li>EfficientNetRandomCrop</li><li>RandomErasing</li><li>EfficientNetCenterCrop</li><li>ResizeEdge</li><li>ColorJitter</li><li>Lighting</li><li>Albumentations</li><li>Albu</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_74e1e thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_74e1e_row0_col0, #T_74e1e_row0_col1, #T_74e1e_row0_col2, #T_74e1e_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_74e1e">
  <thead>
    <tr>
      <th id="T_74e1e_level0_col0" class="col_heading level0 col0" >model (part 1)</th>
      <th id="T_74e1e_level0_col1" class="col_heading level0 col1" >model (part 2)</th>
      <th id="T_74e1e_level0_col2" class="col_heading level0 col2" >model (part 3)</th>
      <th id="T_74e1e_level0_col3" class="col_heading level0 col3" >model (part 4)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_74e1e_row0_col0" class="data row0 col0" ><ul><li>AlexNet</li><li>ShiftWindowMSA</li><li>ClsDataPreprocessor</li><li>VisionTransformer</li><li>BEiT</li><li>Conformer</li><li>ConvMixer</li><li>ResNet</li><li>ResNetV1c</li><li>ResNetV1d</li><li>ResNeXt</li><li>CSPDarkNet</li><li>CSPResNet</li><li>CSPResNeXt</li><li>DaViT</li><li>DistilledVisionTransformer</li><li>DeiT3</li><li>DenseNet</li><li>PoolFormer</li><li>EfficientFormer</li></ul></td>
      <td id="T_74e1e_row0_col1" class="data row0 col1" ><ul><li>EfficientNet</li><li>EfficientNetV2</li><li>HorNet</li><li>HRNet</li><li>InceptionV3</li><li>LeNet5</li><li>MixMIMTransformer</li><li>MlpMixer</li><li>MobileNetV2</li><li>MobileNetV3</li><li>MobileOne</li><li>MViT</li><li>RegNet</li><li>RepLKNet</li><li>RepMLPNet</li><li>RepVGG</li><li>Res2Net</li><li>ResNeSt</li><li>ResNet_CIFAR</li><li>RevVisionTransformer</li></ul></td>
      <td id="T_74e1e_row0_col2" class="data row0 col2" ><ul><li>SEResNet</li><li>SEResNeXt</li><li>ShuffleNetV1</li><li>ShuffleNetV2</li><li>SwinTransformer</li><li>SwinTransformerV2</li><li>T2T_ViT</li><li>TIMMBackbone</li><li>TNT</li><li>PCPVT</li><li>SVT</li><li>VAN</li><li>VGG</li><li>HuggingFaceClassifier</li><li>ImageClassifier</li><li>TimmClassifier</li><li>ClsHead</li><li>ConformerHead</li><li>VisionTransformerClsHead</li><li>DeiTClsHead</li></ul></td>
      <td id="T_74e1e_row0_col3" class="data row0 col3" ><ul><li>EfficientFormerClsHead</li><li>LinearClsHead</li><li>AsymmetricLoss</li><li>CrossEntropyLoss</li><li>FocalLoss</li><li>LabelSmoothLoss</li><li>SeesawLoss</li><li>ArcFaceClsHead</li><li>MultiLabelClsHead</li><li>CSRAClsHead</li><li>MultiLabelLinearClsHead</li><li>MultiTaskHead</li><li>StackedLinearClsHead</li><li>GlobalAveragePooling</li><li>GeneralizedMeanPooling</li><li>HRFuseScales</li><li>LinearReduction</li><li>ImageToImageRetriever</li><li>AverageClsScoreTTA</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>
<details open><div align='center'><b>MMclassification Tools</b></div>
<div align='center'>
<style type="text/css">
#T_ba612 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_ba612_row0_col0, #T_ba612_row0_col1, #T_ba612_row0_col2, #T_ba612_row0_col3, #T_ba612_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_ba612">
  <thead>
    <tr>
      <th id="T_ba612_level0_col0" class="col_heading level0 col0" >tools/misc</th>
      <th id="T_ba612_level0_col1" class="col_heading level0 col1" >tools/visualizations</th>
      <th id="T_ba612_level0_col2" class="col_heading level0 col2" >tools/torchserve</th>
      <th id="T_ba612_level0_col3" class="col_heading level0 col3" >.dev_scripts</th>
      <th id="T_ba612_level0_col4" class="col_heading level0 col4" >tools/analysis_tools</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_ba612_row0_col0" class="data row0 col0" ><ul><li>verify_dataset.py</li><li>print_config.py</li></ul></td>
      <td id="T_ba612_row0_col1" class="data row0 col1" ><ul><li>browse_dataset.py</li><li>vis_scheduler.py</li><li>vis_cam.py</li></ul></td>
      <td id="T_ba612_row0_col2" class="data row0 col2" ><ul><li>mmcls_handler.py</li><li>mmcls2torchserve.py</li><li>test_torchserver.py</li></ul></td>
      <td id="T_ba612_row0_col3" class="data row0 col3" ><ul><li>compare_init.py</li><li>ckpt_tree.py</li><li>generate_readme.py</li></ul></td>
      <td id="T_ba612_row0_col4" class="data row0 col4" ><ul><li>eval_metric.py</li><li>analyze_results.py</li><li>analyze_logs.py</li><li>get_flops.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_3bd40 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_3bd40_row0_col0, #T_3bd40_row0_col1, #T_3bd40_row0_col2, #T_3bd40_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_3bd40">
  <thead>
    <tr>
      <th id="T_3bd40_level0_col0" class="col_heading level0 col0" >.dev_scripts/benchmark_regression</th>
      <th id="T_3bd40_level0_col1" class="col_heading level0 col1" >tools</th>
      <th id="T_3bd40_level0_col2" class="col_heading level0 col2" >tools/model_converters (part 1)</th>
      <th id="T_3bd40_level0_col3" class="col_heading level0 col3" >tools/model_converters (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_3bd40_row0_col0" class="data row0 col0" ><ul><li>bench_train.yml</li><li>4-benchmark_speed.py</li><li>3-benchmark_train.py</li><li>1-benchmark_valid.py</li><li>2-benchmark_test.py</li></ul></td>
      <td id="T_3bd40_row0_col1" class="data row0 col1" ><ul><li>dist_test.sh</li><li>slurm_test.sh</li><li>test.py</li><li>dist_train.sh</li><li>train.py</li><li>slurm_train.sh</li><li>kfold-cross-valid.py</li></ul></td>
      <td id="T_3bd40_row0_col2" class="data row0 col2" ><ul><li>efficientnet_to_mmcls.py</li><li>repvgg_to_mmcls.py</li><li>clip_to_mmcls.py</li><li>reparameterize_model.py</li><li>shufflenetv2_to_mmcls.py</li><li>van2mmcls.py</li><li>hornet2mmcls.py</li><li>mixmimx_to_mmcls.py</li><li>edgenext_to_mmcls.py</li><li>torchvision_to_mmcls.py</li><li>twins2mmcls.py</li><li>revvit_to_mmcls.py</li></ul></td>
      <td id="T_3bd40_row0_col3" class="data row0 col3" ><ul><li>convnext_to_mmcls.py</li><li>replknet_to_mmcls.py</li><li>efficientnetv2_to_mmcls.py</li><li>mobilenetv2_to_mmcls.py</li><li>mlpmixer_to_mmcls.py</li><li>davit_to_mmcls.py</li><li>vgg_to_mmcls.py</li><li>deit3_to_mmcls.py</li><li>eva_to_mmcls.py</li><li>publish_model.py</li><li>tinyvit_to_mmcls.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>

## MMsegmentation (1.0.0rc5)

<details open><div align='center'><b>MMsegmentation Module Components</b></div>
<div align='center'>
<style type="text/css">
#T_aa436 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_aa436_row0_col0, #T_aa436_row0_col1, #T_aa436_row0_col2, #T_aa436_row0_col3, #T_aa436_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_aa436">
  <thead>
    <tr>
      <th id="T_aa436_level0_col0" class="col_heading level0 col0" >task util</th>
      <th id="T_aa436_level0_col1" class="col_heading level0 col1" >visualizer</th>
      <th id="T_aa436_level0_col2" class="col_heading level0 col2" >hook</th>
      <th id="T_aa436_level0_col3" class="col_heading level0 col3" >optimizer wrapper constructor</th>
      <th id="T_aa436_level0_col4" class="col_heading level0 col4" >metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_aa436_row0_col0" class="data row0 col0" ><ul><li>OHEMPixelSampler</li></ul></td>
      <td id="T_aa436_row0_col1" class="data row0 col1" ><ul><li>SegLocalVisualizer</li></ul></td>
      <td id="T_aa436_row0_col2" class="data row0 col2" ><ul><li>SegVisualizationHook</li></ul></td>
      <td id="T_aa436_row0_col3" class="data row0 col3" ><ul><li>LearningRateDecayOptimizerConstructor</li><li>LayerDecayOptimizerConstructor</li></ul></td>
      <td id="T_aa436_row0_col4" class="data row0 col4" ><ul><li>CitysMetric</li><li>IoUMetric</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_f41af thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_f41af_row0_col0, #T_f41af_row0_col1, #T_f41af_row0_col2, #T_f41af_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_f41af">
  <thead>
    <tr>
      <th id="T_f41af_level0_col0" class="col_heading level0 col0" >dataset (part 1)</th>
      <th id="T_f41af_level0_col1" class="col_heading level0 col1" >dataset (part 2)</th>
      <th id="T_f41af_level0_col2" class="col_heading level0 col2" >transform (part 1)</th>
      <th id="T_f41af_level0_col3" class="col_heading level0 col3" >transform (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_f41af_row0_col0" class="data row0 col0" ><ul><li>BaseSegDataset</li><li>ADE20KDataset</li><li>ChaseDB1Dataset</li><li>CityscapesDataset</li><li>COCOStuffDataset</li><li>DarkZurichDataset</li><li>MultiImageMixDataset</li><li>DecathlonDataset</li><li>DRIVEDataset</li><li>HRFDataset</li><li>iSAIDDataset</li></ul></td>
      <td id="T_f41af_row0_col1" class="data row0 col1" ><ul><li>ISPRSDataset</li><li>LIPDataset</li><li>LoveDADataset</li><li>NightDrivingDataset</li><li>PascalContextDataset</li><li>PascalContextDataset59</li><li>PotsdamDataset</li><li>STAREDataset</li><li>SynapseDataset</li><li>PascalVOCDataset</li></ul></td>
      <td id="T_f41af_row0_col2" class="data row0 col2" ><ul><li>PackSegInputs</li><li>LoadAnnotations</li><li>LoadImageFromNDArray</li><li>LoadBiomedicalImageFromFile</li><li>LoadBiomedicalAnnotation</li><li>LoadBiomedicalData</li><li>ResizeToMultiple</li><li>Rerange</li><li>CLAHE</li><li>RandomCrop</li><li>RandomRotate</li><li>RGB2Gray</li><li>AdjustGamma</li></ul></td>
      <td id="T_f41af_row0_col3" class="data row0 col3" ><ul><li>SegRescale</li><li>PhotoMetricDistortion</li><li>RandomCutOut</li><li>RandomRotFlip</li><li>RandomMosaic</li><li>GenerateEdge</li><li>ResizeShortestEdge</li><li>BioMedical3DRandomCrop</li><li>BioMedicalGaussianNoise</li><li>BioMedicalGaussianBlur</li><li>BioMedicalRandomGamma</li><li>BioMedical3DPad</li><li>BioMedical3DRandomFlip</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_d6c05 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_d6c05_row0_col0, #T_d6c05_row0_col1, #T_d6c05_row0_col2, #T_d6c05_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_d6c05">
  <thead>
    <tr>
      <th id="T_d6c05_level0_col0" class="col_heading level0 col0" >model (part 1)</th>
      <th id="T_d6c05_level0_col1" class="col_heading level0 col1" >model (part 2)</th>
      <th id="T_d6c05_level0_col2" class="col_heading level0 col2" >model (part 3)</th>
      <th id="T_d6c05_level0_col3" class="col_heading level0 col3" >model (part 4)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_d6c05_row0_col0" class="data row0 col0" ><ul><li>VisionTransformer</li><li>BEiT</li><li>BiSeNetV1</li><li>BiSeNetV2</li><li>CGNet</li><li>ERFNet</li><li>CrossEntropyLoss</li><li>DiceLoss</li><li>FocalLoss</li><li>LovaszLoss</li><li>TverskyLoss</li><li>ANNHead</li><li>APCHead</li><li>ASPPHead</li><li>FCNHead</li><li>CCHead</li><li>DAHead</li><li>DMHead</li><li>DNLHead</li></ul></td>
      <td id="T_d6c05_row0_col1" class="data row0 col1" ><ul><li>DPTHead</li><li>EMAHead</li><li>EncHead</li><li>FPNHead</li><li>GCHead</li><li>ISAHead</li><li>KernelUpdator</li><li>KernelUpdateHead</li><li>IterativeDecodeHead</li><li>LRASPPHead</li><li>Mask2FormerHead</li><li>MaskFormerHead</li><li>NLHead</li><li>OCRHead</li><li>PointHead</li><li>PSAHead</li><li>PSPHead</li><li>SegformerHead</li><li>SegmenterMaskTransformerHead</li></ul></td>
      <td id="T_d6c05_row0_col2" class="data row0 col2" ><ul><li>DepthwiseSeparableASPPHead</li><li>DepthwiseSeparableFCNHead</li><li>SETRMLAHead</li><li>SETRUPHead</li><li>STDCHead</li><li>UPerHead</li><li>FastSCNN</li><li>ResNet</li><li>ResNetV1c</li><li>ResNetV1d</li><li>HRNet</li><li>ICNet</li><li>MAE</li><li>MixVisionTransformer</li><li>MobileNetV2</li><li>MobileNetV3</li><li>ResNeSt</li><li>ResNeXt</li><li>STDCNet</li></ul></td>
      <td id="T_d6c05_row0_col3" class="data row0 col3" ><ul><li>STDCContextPathNet</li><li>SwinTransformer</li><li>TIMMBackbone</li><li>PCPVT</li><li>SVT</li><li>DeconvModule</li><li>InterpConv</li><li>UNet</li><li>SegDataPreProcessor</li><li>Feature2Pyramid</li><li>FPN</li><li>ICNeck</li><li>JPU</li><li>MLANeck</li><li>MultiLevelNeck</li><li>EncoderDecoder</li><li>CascadeEncoderDecoder</li><li>SegTTAModel</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>
<details open><div align='center'><b>MMsegmentation Tools</b></div>
<div align='center'>
<style type="text/css">
#T_2c057 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_2c057_row0_col0, #T_2c057_row0_col1, #T_2c057_row0_col2, #T_2c057_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_2c057">
  <thead>
    <tr>
      <th id="T_2c057_level0_col0" class="col_heading level0 col0" >tools/deployment</th>
      <th id="T_2c057_level0_col1" class="col_heading level0 col1" >tools/misc</th>
      <th id="T_2c057_level0_col2" class="col_heading level0 col2" >tools/torchserve</th>
      <th id="T_2c057_level0_col3" class="col_heading level0 col3" >tools/analysis_tools</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_2c057_row0_col0" class="data row0 col0" ><ul><li>pytorch2torchscript.py</li></ul></td>
      <td id="T_2c057_row0_col1" class="data row0 col1" ><ul><li>browse_dataset.py</li><li>publish_model.py</li><li>print_config.py</li></ul></td>
      <td id="T_2c057_row0_col2" class="data row0 col2" ><ul><li>mmseg_handler.py</li><li>mmseg2torchserve.py</li><li>test_torchserve.py</li></ul></td>
      <td id="T_2c057_row0_col3" class="data row0 col3" ><ul><li>benchmark.py</li><li>confusion_matrix.py</li><li>analyze_logs.py</li><li>get_flops.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_bf7a6 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_bf7a6_row0_col0, #T_bf7a6_row0_col1, #T_bf7a6_row0_col2 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_bf7a6">
  <thead>
    <tr>
      <th id="T_bf7a6_level0_col0" class="col_heading level0 col0" >tools</th>
      <th id="T_bf7a6_level0_col1" class="col_heading level0 col1" >tools/model_converters</th>
      <th id="T_bf7a6_level0_col2" class="col_heading level0 col2" >tools/dataset_converters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_bf7a6_row0_col0" class="data row0 col0" ><ul><li>dist_test.sh</li><li>slurm_test.sh</li><li>test.py</li><li>dist_train.sh</li><li>train.py</li><li>slurm_train.sh</li></ul></td>
      <td id="T_bf7a6_row0_col1" class="data row0 col1" ><ul><li>swin2mmseg.py</li><li>vitjax2mmseg.py</li><li>twins2mmseg.py</li><li>stdc2mmseg.py</li><li>vit2mmseg.py</li><li>mit2mmseg.py</li><li>beit2mmseg.py</li></ul></td>
      <td id="T_bf7a6_row0_col2" class="data row0 col2" ><ul><li>voc_aug.py</li><li>hrf.py</li><li>drive.py</li><li>pascal_context.py</li><li>vaihingen.py</li><li>stare.py</li><li>synapse.py</li><li>isaid.py</li><li>cityscapes.py</li><li>loveda.py</li><li>potsdam.py</li><li>chase_db1.py</li><li>coco_stuff164k.py</li><li>coco_stuff10k.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>

## MMengine (0.6.0)

<details open><div align='center'><b>MMengine Module Components</b></div>
<div align='center'>
<style type="text/css">
#T_41b5b thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_41b5b_row0_col0, #T_41b5b_row0_col1, #T_41b5b_row0_col2, #T_41b5b_row0_col3, #T_41b5b_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_41b5b">
  <thead>
    <tr>
      <th id="T_41b5b_level0_col0" class="col_heading level0 col0" >log_processor</th>
      <th id="T_41b5b_level0_col1" class="col_heading level0 col1" >visualizer</th>
      <th id="T_41b5b_level0_col2" class="col_heading level0 col2" >metric</th>
      <th id="T_41b5b_level0_col3" class="col_heading level0 col3" >evaluator</th>
      <th id="T_41b5b_level0_col4" class="col_heading level0 col4" >runner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_41b5b_row0_col0" class="data row0 col0" ><ul><li>LogProcessor</li></ul></td>
      <td id="T_41b5b_row0_col1" class="data row0 col1" ><ul><li>Visualizer</li></ul></td>
      <td id="T_41b5b_row0_col2" class="data row0 col2" ><ul><li>DumpResults</li></ul></td>
      <td id="T_41b5b_row0_col3" class="data row0 col3" ><ul><li>Evaluator</li></ul></td>
      <td id="T_41b5b_row0_col4" class="data row0 col4" ><ul><li>Runner</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_f32ce thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_f32ce_row0_col0, #T_f32ce_row0_col1, #T_f32ce_row0_col2, #T_f32ce_row0_col3, #T_f32ce_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_f32ce">
  <thead>
    <tr>
      <th id="T_f32ce_level0_col0" class="col_heading level0 col0" >optimizer wrapper constructor</th>
      <th id="T_f32ce_level0_col1" class="col_heading level0 col1" >Collate Functions</th>
      <th id="T_f32ce_level0_col2" class="col_heading level0 col2" >data sampler</th>
      <th id="T_f32ce_level0_col3" class="col_heading level0 col3" >vis_backend</th>
      <th id="T_f32ce_level0_col4" class="col_heading level0 col4" >dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_f32ce_row0_col0" class="data row0 col0" ><ul><li>DefaultOptimWrapperConstructor</li></ul></td>
      <td id="T_f32ce_row0_col1" class="data row0 col1" ><ul><li>pseudo_collate</li><li>default_collate</li></ul></td>
      <td id="T_f32ce_row0_col2" class="data row0 col2" ><ul><li>DefaultSampler</li><li>InfiniteSampler</li></ul></td>
      <td id="T_f32ce_row0_col3" class="data row0 col3" ><ul><li>LocalVisBackend</li><li>WandbVisBackend</li><li>TensorboardVisBackend</li></ul></td>
      <td id="T_f32ce_row0_col4" class="data row0 col4" ><ul><li>ConcatDataset</li><li>RepeatDataset</li><li>ClassBalancedDataset</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_467b9 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_467b9_row0_col0, #T_467b9_row0_col1, #T_467b9_row0_col2, #T_467b9_row0_col3, #T_467b9_row0_col4 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_467b9">
  <thead>
    <tr>
      <th id="T_467b9_level0_col0" class="col_heading level0 col0" >optim_wrapper</th>
      <th id="T_467b9_level0_col1" class="col_heading level0 col1" >loop</th>
      <th id="T_467b9_level0_col2" class="col_heading level0 col2" >model_wrapper</th>
      <th id="T_467b9_level0_col3" class="col_heading level0 col3" >model</th>
      <th id="T_467b9_level0_col4" class="col_heading level0 col4" >weight initializer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_467b9_row0_col0" class="data row0 col0" ><ul><li>OptimWrapper</li><li>AmpOptimWrapper</li><li>ApexOptimWrapper</li></ul></td>
      <td id="T_467b9_row0_col1" class="data row0 col1" ><ul><li>EpochBasedTrainLoop</li><li>IterBasedTrainLoop</li><li>ValLoop</li><li>TestLoop</li></ul></td>
      <td id="T_467b9_row0_col2" class="data row0 col2" ><ul><li>DistributedDataParallel</li><li>DataParallel</li><li>MMDistributedDataParallel</li><li>MMSeparateDistributedDataParallel</li></ul></td>
      <td id="T_467b9_row0_col3" class="data row0 col3" ><ul><li>StochasticWeightAverage</li><li>ExponentialMovingAverage</li><li>MomentumAnnealingEMA</li><li>BaseDataPreprocessor</li><li>ImgDataPreprocessor</li><li>BaseTTAModel</li><li>ToyModel</li></ul></td>
      <td id="T_467b9_row0_col4" class="data row0 col4" ><ul><li>Constant</li><li>Xavier</li><li>Normal</li><li>TruncNormal</li><li>Uniform</li><li>Kaiming</li><li>Caffe2Xavier</li><li>Pretrained</li></ul></td>
    </tr>
  </tbody>
</table>
</div><div align='center'>
<style type="text/css">
#T_d5b59 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_d5b59_row0_col0, #T_d5b59_row0_col1, #T_d5b59_row0_col2, #T_d5b59_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_d5b59">
  <thead>
    <tr>
      <th id="T_d5b59_level0_col0" class="col_heading level0 col0" >hook</th>
      <th id="T_d5b59_level0_col1" class="col_heading level0 col1" >optimizer</th>
      <th id="T_d5b59_level0_col2" class="col_heading level0 col2" >parameter scheduler (part 1)</th>
      <th id="T_d5b59_level0_col3" class="col_heading level0 col3" >parameter scheduler (part 2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_d5b59_row0_col0" class="data row0 col0" ><ul><li>CheckpointHook</li><li>EMAHook</li><li>EmptyCacheHook</li><li>IterTimerHook</li><li>LoggerHook</li><li>NaiveVisualizationHook</li><li>ParamSchedulerHook</li><li>ProfilerHook</li><li>NPUProfilerHook</li><li>RuntimeInfoHook</li><li>DistSamplerSeedHook</li><li>SyncBuffersHook</li><li>PrepareTTAHook</li></ul></td>
      <td id="T_d5b59_row0_col1" class="data row0 col1" ><ul><li>ASGD</li><li>Adadelta</li><li>Adagrad</li><li>Adam</li><li>AdamW</li><li>Adamax</li><li>LBFGS</li><li>Optimizer</li><li>RMSprop</li><li>Rprop</li><li>SGD</li><li>SparseAdam</li><li>ZeroRedundancyOptimizer</li></ul></td>
      <td id="T_d5b59_row0_col2" class="data row0 col2" ><ul><li>StepParamScheduler</li><li>MultiStepParamScheduler</li><li>ConstantParamScheduler</li><li>ExponentialParamScheduler</li><li>CosineAnnealingParamScheduler</li><li>LinearParamScheduler</li><li>PolyParamScheduler</li><li>OneCycleParamScheduler</li><li>CosineRestartParamScheduler</li><li>ReduceOnPlateauParamScheduler</li><li>ConstantLR</li><li>CosineAnnealingLR</li><li>ExponentialLR</li><li>LinearLR</li><li>MultiStepLR</li></ul></td>
      <td id="T_d5b59_row0_col3" class="data row0 col3" ><ul><li>StepLR</li><li>PolyLR</li><li>OneCycleLR</li><li>CosineRestartLR</li><li>ReduceOnPlateauLR</li><li>ConstantMomentum</li><li>CosineAnnealingMomentum</li><li>ExponentialMomentum</li><li>LinearMomentum</li><li>MultiStepMomentum</li><li>StepMomentum</li><li>PolyMomentum</li><li>CosineRestartMomentum</li><li>ReduceOnPlateauMomentum</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>

## MMCV (2.0.0rc4)

<details open><div align='center'><b>MMCV Module Components</b></div>
<div align='center'>
<style type="text/css">
#T_be596 thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_be596_row0_col0, #T_be596_row0_col1, #T_be596_row0_col2, #T_be596_row0_col3 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_be596">
  <thead>
    <tr>
      <th id="T_be596_level0_col0" class="col_heading level0 col0" >transform</th>
      <th id="T_be596_level0_col1" class="col_heading level0 col1" >model (part 1)</th>
      <th id="T_be596_level0_col2" class="col_heading level0 col2" >model (part 2)</th>
      <th id="T_be596_level0_col3" class="col_heading level0 col3" >model (part 3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_be596_row0_col0" class="data row0 col0" ><ul><li>LoadImageFromFile</li><li>LoadAnnotations</li><li>Compose</li><li>KeyMapper</li><li>TransformBroadcaster</li><li>RandomChoice</li><li>RandomApply</li><li>Normalize</li><li>Resize</li><li>Pad</li><li>CenterCrop</li><li>RandomGrayscale</li><li>MultiScaleFlipAug</li><li>TestTimeAug</li><li>RandomChoiceResize</li><li>RandomFlip</li><li>RandomResize</li><li>ToTensor</li><li>ImageToTensor</li></ul></td>
      <td id="T_be596_row0_col1" class="data row0 col1" ><ul><li>ReLU</li><li>LeakyReLU</li><li>PReLU</li><li>RReLU</li><li>ReLU6</li><li>ELU</li><li>Sigmoid</li><li>Tanh</li><li>SiLU</li><li>Clamp</li><li>Clip</li><li>GELU</li><li>ContextBlock</li><li>Conv1d</li><li>Conv2d</li><li>Conv3d</li><li>Conv</li><li>Conv2dAdaptivePadding</li></ul></td>
      <td id="T_be596_row0_col2" class="data row0 col2" ><ul><li>BN</li><li>BN1d</li><li>BN2d</li><li>BN3d</li><li>SyncBN</li><li>GN</li><li>LN</li><li>IN</li><li>IN1d</li><li>IN2d</li><li>IN3d</li><li>zero</li><li>reflect</li><li>replicate</li><li>ConvModule</li><li>ConvWS</li><li>ConvAWS</li><li>DropPath</li></ul></td>
      <td id="T_be596_row0_col3" class="data row0 col3" ><ul><li>Dropout</li><li>GeneralizedAttention</li><li>HSigmoid</li><li>HSwish</li><li>NonLocal2d</li><li>Swish</li><li>nearest</li><li>bilinear</li><li>pixel_shuffle</li><li>deconv</li><li>ConvTranspose2d</li><li>deconv3d</li><li>ConvTranspose3d</li><li>MultiheadAttention</li><li>FFN</li><li>BaseTransformerLayer</li><li>TransformerLayerSequence</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>
<details open><div align='center'><b>MMCV Tools</b></div>
<div align='center'>
<style type="text/css">
#T_ea8ce thead th {
  align: center;
  text-align: center;
  vertical-align: bottom;
}
#T_ea8ce_row0_col0 {
  text-align: left;
  align: center;
  vertical-align: top;
}
</style>
<table id="T_ea8ce">
  <thead>
    <tr>
      <th id="T_ea8ce_level0_col0" class="col_heading level0 col0" >.dev_scripts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_ea8ce_row0_col0" class="data row0 col0" ><ul><li>check_installation.py</li></ul></td>
    </tr>
  </tbody>
</table>
</div></details>
