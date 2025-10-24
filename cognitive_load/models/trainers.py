import torch
import torch.nn as nn
import lightning.pytorch as pl

# Optimized 1D CNN Model for 3600-length raw signals
class CNN1D(pl.LightningModule):
    def __init__(self, in_channels=1, num_classes=2, learning_rate=0.0005):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.features = nn.Sequential(
            # First layer: larger kernel to capture broader patterns
            nn.Conv1d(in_channels, 32, kernel_size=51, stride=3, padding=25),  # Increased kernel size
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),  # Using LeakyReLU to avoid dead neurons
            nn.MaxPool1d(8),  # More aggressive pooling for longer signals
            nn.Dropout(0.2),  # Adding early dropout
            
            # Second layer
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(6),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Conv1d(64, 128, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(6),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Add weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# Optimized LSTM Model for 3600-length raw signals
class LSTMModel(pl.LightningModule):
    def __init__(self, in_channels=1, num_classes=2, hidden_size=128, num_layers=2, learning_rate=0.0005):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Feature extractor with more aggressive downsampling for long signals
        self.feature_extractor = nn.Sequential(
            # First layer with larger kernel and stride
            nn.Conv1d(in_channels, 32, kernel_size=51, stride=5, padding=25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(8),
            
            # Second layer
            nn.Conv1d(32, 64, kernel_size=25, stride=3, padding=12),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Dropout(0.3)
        )
        
        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=64,  # Output channels from feature extractor
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Extract features and reduce sequence length
        x = self.feature_extractor(x)  # [batch, channels, seq_len]
        
        # Reshape for LSTM: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        outputs, (h_n, _) = self.lstm(x)
        
        # Concatenate the final forward and backward hidden states
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Classify
        x = self.classifier(h_n)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# Optimized ResNet1D Model for 3600-length raw signals
class ResNet1DModel(pl.LightningModule):
    def __init__(self, in_channels=1, num_classes=2, learning_rate=0.0005):
        super().__init__()
        self.learning_rate = learning_rate
        
        # First conv block to reduce dimensionality - significantly more aggressive for long signals
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=51, stride=6, padding=25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(8)
        )
        
        # ResNet Blocks with optimized parameters for long signals
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(32, 32),
            self._make_residual_block(32, 64, downsample=True),
            self._make_residual_block(64, 64),
            self._make_residual_block(64, 128, downsample=True),
        ])
        
        # Final layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def _make_residual_block(self, in_channels, out_channels, downsample=False):
        layers = []
        stride = 2 if downsample else 1
        
        # Downsample if needed
        if downsample or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        # Main branch with larger kernels for long signals
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=15, stride=1, padding=7, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        return nn.ModuleDict({
            'main': nn.Sequential(*layers),
            'shortcut': shortcut
        })
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            identity = x
            x = block['main'](x)
            x += block['shortcut'](identity)
            x = torch.nn.functional.leaky_relu(x, 0.2)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }