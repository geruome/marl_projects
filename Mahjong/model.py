import torch
from torch import nn
import torch.nn.functional as F


class InitCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # --- Shared Convolutional Block for All 6 Suit Rows (Man, Pin, Sou: Hand/Pack) ---
        # Input to this block will be (Batch_size * 6, 1, 9)
        # It processes each 1x9 row independently, leveraging shared weights.
        self.suit_conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(True),
            # Global Average Pooling to reduce the 9-length dimension to 1
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten() # Flattens to (Batch_size * 6, 32)
        )
        
        # --- MLP Block for 7-column Honor Tiles ---
        # Input to this MLP will be (Batch_size * 2, 7)
        # Each of the 7-feature vectors (for hand or pack) will be mapped to 32 features independently.
        self.honor_mlp_block = nn.Sequential(
            nn.Linear(7, 32), # Input 7 features, directly map to 32 features
            nn.ReLU(True),
            nn.Linear(32, 32), # Optional: Add another layer if needed for complexity
            nn.ReLU(True)
        )

        # --- Final Feature Fusion and Value Prediction Head ---
        # Output from suit_conv_block: (Batch_size, 6 * 32) = (Batch_size, 192)
        # Output from honor_mlp_block: (Batch_size, 2, 32) -> after flatten for concat: (Batch_size, 2 * 32) = (Batch_size, 64)
        # Total combined features = 192 + 64 = 256
        
        self.value_head = nn.Sequential(
            nn.Linear(6 * 32 + 2 * 32, 128), # Total 256 features
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1) # Single value output
        )
        
        # --- Weight Initialization ---
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tensor: torch.tensor):
        obs = tensor.float() # obs shape: (Batch_size, 8, 9)

        # --- Process 6 Suit Rows (Man, Pin, Sou: Hand/Pack) ---
        suit_inputs = obs[:, 0:6, :] # (Batch_size, 6, 9)

        # Reshape for Conv1d: (Batch_size * 6, 1, 9)
        reshaped_suit_inputs = suit_inputs.reshape(-1, 1, 9) 

        # Process through shared Conv1d block with Global Average Pooling
        suit_features_per_row = self.suit_conv_block(reshaped_suit_inputs) # (Batch_size * 6, 32)

        # Reshape back to (Batch_size, 6 * 32) for concatenation
        combined_suit_features = suit_features_per_row.view(obs.shape[0], -1) # (Batch_size, 192)
        
        # --- Process 2 Honor Rows (Hand/Pack) ---
        honor_inputs = obs[:, 6:8, :] # (Batch_size, 2, 9)

        # Discard the last 2 columns, resulting in (Batch_size, 2, 7)
        honor_trimmed = honor_inputs[:, :, :7] 

        # Reshape for MLP: (Batch_size * 2, 7) to apply MLP to each (hand/pack) row independently
        reshaped_honor_input = honor_trimmed.reshape(-1, 7) 
        
        # Process through MLP
        honor_feature_per_row = self.honor_mlp_block(reshaped_honor_input) # (Batch_size * 2, 32)
        
        # Reshape back to (Batch_size, 2, 32) and then flatten the 2 and 32 dimensions for concatenation
        combined_honor_features = honor_feature_per_row.view(obs.shape[0], -1) # (Batch_size, 2 * 32 = 64)
        
        # --- Concatenate all features ---
        combined_all_features = torch.cat([combined_suit_features, combined_honor_features], dim=1)
        
        # --- Predict Value ---
        value = self.value_head(combined_all_features)

        # Return only value
        return value
    

if __name__ == '__main__':
    model = MyModel()
    B = 11
    x = torch.rand(B, 8, 9)
    y = model(x)
    print(y.shape)