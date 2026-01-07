from .pose_hrnet import PoseHighResolutionNet
from .pose_resnet import PoseResNet, BasicBlock, Bottleneck


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(args, **kwargs):
    if args.key_model == 'pose_hrnet':
        model = PoseHighResolutionNet(args, **kwargs)

    if 'resnet' in args.key_model:
        num_layers = ''
        for c in args.key_model:
            if c.isdigit():
                num_layers += c
        block_class, layers = resnet_spec[int(num_layers)]
        model = PoseResNet(block_class, layers, args, **kwargs)

    if args.key_ckpt:
        model.init_weights(args.key_ckpt)

    return model
