import copy
import jax.numpy as np
from flax import linen as nn
from functools import partial
from typing import Any, Callable, Tuple

DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        }, {
            "kernel_size": (3, 3),
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        }, {
            "conv_type": 1,
            "expand_ratio": 4,
            "input_filters": 48,
            "kernel_size": (3, 3),
            "num_repeat": 4,
            "output_filters": 64,
            "se_ratio": 0,
            "strides": 2,
        }, {
            "conv_type": 0,
            "expand_ratio": 4,
            "input_filters": 64,
            "kernel_size": (3, 3),
            "num_repeat": 6,
            "output_filters": 128,
            "se_ratio": 0.25,
            "strides": 2,
        }, {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 128,
            "kernel_size": (3, 3),
            "num_repeat": 9,
            "output_filters": 160,
            "se_ratio": 0.25,
            "strides": 1,
        }, {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 160,
            "kernel_size": (3, 3),
            "num_repeat": 15,
            "output_filters": 256,
            "se_ratio": 0.25,
            "strides": 2,
        }
    ],
    "efficientnetv2-m": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": (3, 3),
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": (3, 3),
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = nn.initializers.variance_scaling(scale=2.0, mode='fan_out', distribution='truncated_normal')

ModuleDef = Any


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(np.ceil(depth_coefficient * repeats))


class MBConvBlock(nn.Module):
    conv: ModuleDef = partial(
        nn.Conv,
        kernel_size=(1, 1),
        strides=1,
        padding='SAME',
        use_bias=False,
        kernel_init=CONV_KERNEL_INITIALIZER,
    )
    norm: ModuleDef = partial(
        nn.BatchNorm,
        use_running_average=True,
        axis=-1,
        momentum=0.9,
    )
    act: Callable = nn.activation.swish
    input_filters: int = 32
    output_filters: int = 16
    expand_ratio: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    se_ratio: float = 0.0
    dropout_rate: float = 0.2
    name: str = None

    @nn.compact
    def __call__(self, x):
        residual = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = self.conv(
                features=filters,
                name=self.name + "expand_conv",
            )(x)
            x = self.norm(
                name=self.name + "expand_bn"
            )(x)
            x = self.act(x)

        # Depthwise conv
        x = self.conv(
            features=x.shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides,
            feature_group_count=x.shape[-1],
            name=self.name + 'dwconv'
        )(x)
        x = self.norm(
            name=self.name + "bn"
        )(x)
        x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = nn.avg_pool(x, x.shape[1:3])

            se = self.conv(
                features=filters_se,
                padding='SAME',
                use_bias=True,
                name=self.name + 'se_reduce'
            )(se)
            se = self.act(se)
            se = self.conv(
                features=filters,
                use_bias=True,
                name=self.name + 'se_expand'
            )(se)
            se = nn.activation.sigmoid(se)

            x = x * se

        # Output phase
        x = self.conv(
            features=self.output_filters,
            name=self.name + 'project_conv'
        )(x)
        x = self.norm(
            name=self.name + "project_bn"
        )(x)

        # Residual
        if (self.strides == 1) and (self.input_filters == self.output_filters):
            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=True,
                    name=self.name + 'drop'
                )(x)
            x = x + residual

        return x


class FusedMBConvBlock(nn.Module):
    conv: ModuleDef = partial(
        nn.Conv,
        kernel_size=(1, 1),
        strides=1,
        padding='SAME',
        use_bias=False,
        kernel_init=CONV_KERNEL_INITIALIZER,
    )
    norm: ModuleDef = partial(
        nn.BatchNorm,
        use_running_average=True,
        axis=-1,
        momentum=0.9,
    )
    act: Callable = nn.activation.swish
    input_filters: int = 32
    output_filters: int = 16
    expand_ratio: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    se_ratio: float = 0.0
    dropout_rate: float = 0.2
    name: str = None

    @nn.compact
    def __call__(self, x):
        residual = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = self.conv(
                features=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                name=self.name + "expand_conv",
            )(x)
            x = self.norm(
                name=self.name + "expand_bn"
            )(x)
            x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = nn.avg_pool(x, x.shape[1:3])

            se = self.conv(
                features=filters_se,
                use_bias=True,
                name=self.name + 'se_reduce'
            )(se)
            se = self.act(se)
            se = self.conv(
                features=filters,
                use_bias=True,
                name=self.name + 'se_expand'
            )(se)
            se = nn.activation.sigmoid(se)

            x = x * se

        # Output phase
        x = self.conv(
            features=self.output_filters,
            kernel_size=(1, 1) if self.expand_ratio != 1 else self.kernel_size,
            strides=1 if self.expand_ratio != 1 else self.strides,
            name=self.name + 'project_conv'
        )(x)
        x = self.norm(
            name=self.name + "project_bn"
        )(x)
        if self.expand_ratio == 1:
            x = self.act(x)

        # Residual
        if (self.strides == 1) and (self.input_filters == self.output_filters):
            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=True,
                    name=self.name + 'drop'
                )(x)
            x = x + residual

        return x


class EfficientNetV2(nn.Module):
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.activation.swish
    model_name: str = None
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    default_size: int = 224
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: int = 8
    bn_momentum: float = 0.9
    pooling: str = None

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(
            self.conv,
            kernel_size=(1, 1),
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_init=CONV_KERNEL_INITIALIZER,
        )
        norm = partial(
            self.norm,
            use_running_average=not train,
            axis=-1,
            momentum=self.bn_momentum,
        )

        blocks_args = DEFAULT_BLOCKS_ARGS[self.model_name]

        # Build stem
        stem_filters = round_filters(
            filters=blocks_args[0]["input_filters"],
            width_coefficient=self.width_coefficient,
            min_depth=self.min_depth,
            depth_divisor=self.depth_divisor,
        )
        x = conv(
            features=stem_filters,
            kernel_size=(3, 3),
            strides=2,
            name="stem_conv",
        )(x)
        x = norm(
            name="stem_bn"
        )(x)
        x = self.act(x)

        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args["num_repeat"] for args in blocks_args))

        for (i, args) in enumerate(blocks_args):
            assert args["num_repeat"] > 0

            # Update block input and output filters based on depth multiplier.
            args["input_filters"] = round_filters(
                filters=args["input_filters"],
                width_coefficient=self.width_coefficient,
                min_depth=self.min_depth,
                depth_divisor=self.depth_divisor
            )
            args["output_filters"] = round_filters(
                filters=args["output_filters"],
                width_coefficient=self.width_coefficient,
                min_depth=self.min_depth,
                depth_divisor=self.depth_divisor
            )

            # Determine which conv type to use:
            block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
            repeats = round_repeats(
                repeats=args.pop("num_repeat"), depth_coefficient=self.depth_coefficient)
            for j in range(repeats):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]

                x = block(
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    dropout_rate=self.drop_connect_rate * b / blocks,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                    **args,
                )(x)
                b += 1

            self.sow('intermediates', 'features{}'.format(i + 1), x)

        return x


EfficientNetV2B0 = partial(
    EfficientNetV2,
    model_name="efficientnetv2-b0",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=224,
)


EfficientNetV2B1 = partial(
    EfficientNetV2,
    model_name="efficientnetv2-b1",
    width_coefficient=1.0,
    depth_coefficient=1.1,
    default_size=240,
)


EfficientNetV2B2 = partial(
    EfficientNetV2,
    model_name="efficientnetv2-b2",
    width_coefficient=1.1,
    depth_coefficient=1.2,
    default_size=260,
)


EfficientNetV2B3 = partial(
    EfficientNetV2,
    model_name="efficientnetv2-b3",
    width_coefficient=1.2,
    depth_coefficient=1.4,
    default_size=300,
)


EfficientNetV2S = partial(
    EfficientNetV2,
    model_name="efficientnetv2-s",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=384,
)


EfficientNetV2M = partial(
    EfficientNetV2,
    model_name="efficientnetv2-m",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=480,
)


EfficientNetV2L = partial(
    EfficientNetV2,
    model_name="efficientnetv2-l",
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=480,
)
