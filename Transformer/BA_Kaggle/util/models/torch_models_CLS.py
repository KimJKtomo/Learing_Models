import timm
from torch import nn


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        # CNN
        if self.args.model == 'nasnet':
            self.model = timm.create_model('nasnetalarge', pretrained=True,)
        if self.args.model == 'tf_Effiv2_s':
            self.model = timm.create_model(
                'tf_efficientnetv2_s_in21k', pretrained=True,)
        if self.args.model == 'tf_Effiv4_s':
            self.model = timm.create_model(
                'tf_efficientnetv4_s_in21k', pretrained=True,)
            

        if self.args.model == 'xception':
            self.model = timm.create_model('xception', pretrained=True,)
        if self.args.model == 'rexnet':  # 224
            self.model = timm.create_model('rexnet_200', pretrained=True,)
        if self.args.model == 'resnetv2_50_bitm_21k':
            self.model = timm.create_model(
                'resnetv2_50x1_bitm_in21k', pretrained=True,)
        if self.args.model == 'resnetv2_50':
            self.model = timm.create_model('resnetv2_50', pretrained=True,)
        if self.args.model == 'resnext50':
            self.model = timm.create_model('resnext50_32x4d', pretrained=True,)
        if self.args.model == 'seresnet50':
            self.model = timm.create_model('seresnet50', pretrained=True,)
        if self.args.model == 'seresnext50':
            self.model = timm.create_model(
                'seresnext50_32x4d', pretrained=True,)
        if self.args.model == 'vovnet':
            self.model = timm.create_model('ese_vovnet39b', pretrained=True,)
        if self.args.model == 'tresnet_xl':  # 448
            self.model = timm.create_model('tresnet_xl_448', pretrained=True,)

        # mixer
        if self.args.model == 'mixer_b':
            self.model = timm.create_model('mixer_b16_224', pretrained=True,)
        if self.args.model == 'mixer_b_21k':
            self.model = timm.create_model(
                'mixer_b16_224_in21k', pretrained=True,)
        if self.args.model == 'gmixer_24':
            self.model = timm.create_model('gmixer_24_224', pretrained=True,)
        if self.args.model == 'resmlp_24':
            self.model = timm.create_model('resmlp_24_224', pretrained=True,)
        if self.args.model == 'resmlp_24_distilled':
            self.model = timm.create_model(
                'resmlp_24_distilled_224', pretrained=True,)
        if self.args.model == 'gmlp_s16':
            self.model = timm.create_model('gmlp_s16_224', pretrained=True,)

        # transformer
        if self.args.model == 'vit':
            self.model = timm.create_model(
                'vit_base_patch16_224', pretrained=True,)
        if self.args.model == 'vit_notpre':
            self.model = timm.create_model('vit_base_patch16_224')
        if self.args.model == 'vit_l_p16_384':
            self.model = timm.create_model(
                'vit_large_patch16_384', pretrained=True,)
        if self.args.model == 'vit_l_r50_384':
            self.model = timm.create_model(
                'vit_large_r50_s32_384', pretrained=True,)
        if self.args.model == 'beit_l_224':
            self.model = timm.create_model(
                'beit_large_patch16_224', pretrained=True,)
        if self.args.model == 'beit_l_224_22k':
            self.model = timm.create_model(
                'beit_large_patch16_224_in22k', pretrained=True,)
        if self.args.model == 'deit_b_224':
            self.model = timm.create_model(
                'deit_base_patch16_224', pretrained=True,)
        if self.args.model == 'deit_b_distilled_224':
            self.model = timm.create_model(
                'deit_base_distilled_patch16_224', pretrained=True,)
        if self.args.model == 'SwinT_b_224_21k':
            self.model = timm.create_model(
                'swin_base_patch4_window7_224_in22k', pretrained=True,)
        if self.args.model == 'SwinT_b_384_21k':
            self.model = timm.create_model(
                'swin_base_patch4_window12_384_in22k', pretrained=True,)
        if self.args.model == 'SwinT_l_224_21k':
            self.model = timm.create_model(
                'swin_large_patch4_window7_224_in22k', pretrained=True,)
        if self.args.model == 'SwinT_l_384_21k':
            self.model = timm.create_model(
                'swin_large_patch4_window12_384_in22k', pretrained=True,)
        # if self.args.model == 'SwinTv2_s_224':
        #     self.model = timm.create_model('swin_v2_cr_small_224', pretrained=True,)

        # fusion
        if self.args.model == 'convit_b':
            self.model = timm.create_model('convit_base', pretrained=True,)

        self.model.reset_classifier(args.class_n)

    def forward(self, inputs):
        return self.model(inputs)
