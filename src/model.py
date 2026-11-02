"""CNN mask classifier for the face mask detection project.

This module defines :class:`MaskCNN`, a small Convolutional Neural Network
trained from scratch in PyTorch. It is the *classifier* half of the
two-stage pipeline:

    OpenCV face detector  ->  crop each face  ->  MaskCNN classifier

The model only answers the question "is this cropped face wearing a
mask?". It does **not** detect faces in a full image -- that job is done
by OpenCV.

Expected input
--------------
A batch of cropped face images, already resized to 128 x 128 RGB:

    tensor shape = (batch_size, 3, 128, 128)
    dtype        = float32, normalized to roughly [0, 1] or standard
                   ImageNet-style normalization (decided in the data
                   pipeline, not here)

Expected output
---------------
Raw class logits (NOT probabilities):

    tensor shape = (batch_size, 2)

Channel index 0 corresponds to ``with_mask``, channel index 1 to
``without_mask`` -- the same alphabetical order that
``torchvision.datasets.ImageFolder`` assigns.
"""

from __future__ import annotations

import torch
from torch import nn


HEADS = ("flatten", "gap")


class MaskCNN(nn.Module):
    """A small CNN that classifies a 128x128 face crop as mask / no mask.

    The network has two clearly separated parts:

    * ``self.features``   -- the convolutional feature extractor
                             (4 Conv -> BN -> ReLU -> MaxPool blocks).
    * ``self.classifier`` -- the fully connected (FCNN) classifier head
                             that maps the features to 2 logits.

    Two classifier heads are available:

    * ``head="flatten"`` (original) -- Flatten -> Linear(16384, 256) ->
      ReLU -> Dropout -> Linear(256, 2). About 4.6M parameters, ~90% of
      them in the first Linear layer. Requires a 128x128 input.
    * ``head="gap"`` (edge-friendly) -- global average pooling ->
      Dropout -> Linear(256, 2). About 0.39M parameters, and works with
      any input size, so it can also run on smaller crops such as 96x96.
    """

    def __init__(self, num_classes: int = 2, head: str = "flatten") -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"Unknown head {head!r}; expected one of {HEADS}")
        self.head = head

        # ------------------------------------------------------------------
        # Feature extractor (CNN part)
        # ------------------------------------------------------------------
        # Each "Conv block" follows the same recipe:
        #     Conv2d  -> BatchNorm2d -> ReLU -> MaxPool2d
        #
        # Conv2d uses kernel_size=3 with padding=1 so the spatial size is
        # preserved by the convolution itself. The MaxPool2d (kernel=2,
        # stride=2) is what halves the spatial size at each block.
        #
        # Spatial size flow (H = W shown):
        #   input        : 3   x 128 x 128
        #   after block 1: 32  x  64 x  64
        #   after block 2: 64  x  32 x  32
        #   after block 3: 128 x  16 x  16
        #   after block 4: 256 x   8 x   8
        self.features = nn.Sequential(
            # Conv Block 1: 3 -> 32, spatial 128 -> 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 32 -> 64, spatial 64 -> 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3: 64 -> 128, spatial 32 -> 16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 4: 128 -> 256, spatial 16 -> 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ------------------------------------------------------------------
        # Classifier (FCNN part)
        # ------------------------------------------------------------------
        # After the feature extractor, each image is a 256 x 8 x 8 tensor.
        # We flatten it into a 1D vector of length 256 * 8 * 8 = 16384,
        # then run it through a small fully connected network that ends
        # in `num_classes` logits.
        #
        # Dropout(p=0.5) is a regularizer that randomly zeroes half of
        # the activations during training to reduce overfitting. It is
        # automatically disabled when the model is in eval mode.
        #
        # The "gap" head averages each of the 256 feature maps down to a
        # single number instead of flattening all 8 x 8 positions. That
        # removes the 16384 x 256 weight matrix, which is what makes the
        # original model large and slow on CPU-only edge devices.
        if head == "flatten":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through the network.

        Parameters
        ----------
        x : torch.Tensor
            Float tensor of shape ``(batch_size, 3, 128, 128)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch_size, num_classes)``. Softmax is
            **not** applied here; ``nn.CrossEntropyLoss`` expects logits
            during training, and softmax can be applied at inference time
            when we want probabilities.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model_from_checkpoint(checkpoint: dict) -> MaskCNN:
    """Recreate the architecture a checkpoint was trained with.

    Checkpoints saved before the ``head`` key existed used the original
    flatten head, so that is the fallback.
    """
    return MaskCNN(
        num_classes=len(checkpoint["class_to_idx"]),
        head=checkpoint.get("head", "flatten"),
    )


if __name__ == "__main__":
    # Quick sanity check that wiring + tensor shapes are correct.
    expected = torch.Size([1, 2])
    for head in HEADS:
        model = MaskCNN(num_classes=2, head=head)
        model.eval()  # disables Dropout / uses running stats for BatchNorm

        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(dummy)

        print(f"[{head}] params: {count_parameters(model):,}  "
              f"input: {tuple(dummy.shape)}  output: {tuple(out.shape)}")
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print(f"OK: output shape matches {expected} for every head")
