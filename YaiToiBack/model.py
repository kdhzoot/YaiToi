import torch
import torchvision


def get_model():
    # Backbone 정의
    backbone = torchvision.models.vgg16().features[:-1]
    backbone_out = 512
    backbone.out_channels = backbone_out

    # Anchor Generator 정의
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # RoI Pooler 정의
    resolution = 7
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=resolution, sampling_ratio=2
    )

    # Box Head 및 Box Predictor 정의
    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
        in_channels=backbone_out * (resolution ** 2),
        representation_size=4096
    )
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        4096, 15  # 클래스의 개수
    )

    # Faster R-CNN 모델 정의
    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=None,
        min_size=600, max_size=1000,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=6000, rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
        rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor,
        box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=300,
        box_fg_iou_thresh=0.5, box_be_iou_thresh=0.5,
        box_batch_size_per_image=128, box_positive_fraction=0.25
    )

    return model

    # # RPN 및 ROI Heads 파라미터 초기화
    # for param in self.model.rpn.parameters():
    #     torch.nn.init.normal_(param, mean=0.0, std=0.01)
    #
    # for name, param in self.model.roi_heads.named_parameters():
    #     if "bbox_pred" in name:
    #         torch.nn.init.normal_(param, mean=0.0, std=0.001)
    #     elif "weight" in name:
    #         torch.nn.init.normal_(param, mean=0.0, std=0.01)
    #     if "bias" in name:
    #         torch.nn.init.zeros_(param)
