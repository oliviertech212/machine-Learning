import torch.nn as nn

class ImageClassifier(nn.Module):
    """
    CNN model to classify images into 3 categories:
    laptop, food, bag
    """
    def __init__(self, num_classes=3):
        super().__init__()

        # Feature learning - finds patterns, edges, shapes in images
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),         # 64x64 -> 32x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),         # 32x32 -> 16x16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),         # 16x16 -> 8x8
        )

        # Decision making - classifies what it found
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),          # prevents overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)      # learn the patterns
        x = self.classifier(x)    # make the decision
        return x
