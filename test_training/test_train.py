import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

class GestureFeatureExtractor:
    """Extract meaningful features from CSI info
       Necessary because raw amplitude is too noisy for small dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_temporal_features(self, x):
        """Temporal patterns"""
        #Trailing and leading 0's
        x = np.trim_zeros(x)
        if len(x) < 5:
            return np.zeros(20)  # avoids rows with too little data
        
        features = []
        
        # Basic stats
        features.extend([
            np.mean(x),
            np.std(x),
            np.var(x),
            np.max(x) - np.min(x), 
            skew(x),
            kurtosis(x)
        ])
        
        # Temporal diff (changes between time steps), good for slope/speed
        diff1 = np.diff(x)
        if len(diff1) > 0:
            features.extend([
                np.mean(diff1),
                np.std(diff1),
                np.max(diff1) - np.min(diff1),
                np.sum(diff1**2),  # total
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Second diff (change of the change ^)
        diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0])
        if len(diff2) > 0:
            features.extend([
                np.mean(diff2),
                np.std(diff2),
                np.sum(diff2**2)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Zero crossing (how often signal crosses 0), good for oscillation in gestures like wave
        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)
        features.append(zero_crossings)
        
        # Peaks, finds local maxima and minima (peaks, valleys)
        peaks, _ = scipy_signal.find_peaks(x, height=np.mean(x))
        valleys, _ = scipy_signal.find_peaks(-x, height=-np.mean(x))
        features.extend([
            len(peaks),
            len(valleys),
            len(peaks) + len(valleys)
        ])
        
        # Energy in different segments
        '''
            Divides into 3 segments, sum of squares for each segment
            good for how motion is distributed
        '''
        n = len(x)
        if n >= 6:
            seg1 = x[:n//3]
            seg2 = x[n//3:2*n//3]
            seg3 = x[2*n//3:]
            
            features.extend([
                np.sum(seg1**2),
                np.sum(seg2**2),
                np.sum(seg3**2)
            ])
        else:
            features.extend([0, 0, 0])
            
        # returns a feature vector of the above
        return np.array(features)
    
    def extract_frequency_features(self, x):
        """Frequency domain features"""
        
        #Leading trailing zeros again
        x = np.trim_zeros(x)
        if len(x) < 8:
            return np.zeros(5)
        
        # Fast Fourier Transform (converts from time domain to frequency)
        fft = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x))
        
        # Power spectral density (energy at each frequency)
        psd = np.abs(fft)**2
        
        features = []
        
        # Spectral centroid, weighted average of frequencies in the signal, high centroid -> high frequency -> fast movement
        if np.sum(psd) > 0:
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            features.append(spectral_centroid)
        else:
            features.append(0)
        
        # Spectral energy, total signal energy, higher energy -> stronger movement (good for comparing control to any other gesture)
        features.append(np.sum(psd))
        
        # Dominant frequency, good for most prominent motion
        dominant_freq_idx = np.argmax(psd[:len(psd)//2])
        features.append(freqs[dominant_freq_idx])
        
        # Spectral spread, frequency distribution around centroid, bigger spread -> more complex movement
        mean_freq = features[0]
        if np.sum(psd) > 0:
            spectral_spread = np.sqrt(np.sum(((freqs[:len(freqs)//2] - mean_freq)**2) * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
            features.append(spectral_spread)
        else:
            features.append(0)
        
        # High frequency energy ratio, good for fast motions with sharp changes in frequency
        mid_point = len(psd) // 4
        high_freq_energy = np.sum(psd[mid_point:len(psd)//2])
        total_energy = np.sum(psd[:len(psd)//2])
        features.append(high_freq_energy / total_energy if total_energy > 0 else 0)
        
        return np.array(features)
    
    def extract_pattern_features(self, x):
        """Pattern-based features
        *Better for distinguishing gestures with similar above features
        """
        #Leading trailing zeros again again
        x = np.trim_zeros(x)
        if len(x) < 5:
            return np.zeros(8)
        
        features = []
        
        # Slope analysis, fits a line to signal and can show if it trends in a certain way (increase, decrease)
        if len(x) > 2:
            trend = np.polyfit(range(len(x)), x, 1)[0]
            features.append(trend)
        else:
            features.append(0)
        
        # Self similarity at different lags, good for repitition/rhythmic patterns
        for lag in [1, 2, 3]:
            if len(x) > lag:
                corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
            else:
                features.append(0)
        
        # Hilbert transform to get signal envelope, good for showing how amplitude evolves (soft or sharp movements)
        try:
            analytic_signal = scipy_signal.hilbert(x)
            envelope = np.abs(analytic_signal)
            features.extend([
                np.mean(envelope),
                np.std(envelope),
                np.max(envelope) - np.min(envelope)
            ])
        except:
            features.extend([0, 0, 0])
        
        # RMS (average power)
        rms = np.sqrt(np.mean(x**2))
        features.append(rms)
        
        return np.array(features)
    
    def extract_all_features(self, X):
        """Extract all features for a dataset"""
        all_features = []
        
        for i, x in enumerate(X):
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(X)}")
                
            # Extract the features
            temporal = self.extract_temporal_features(x)
            frequency = self.extract_frequency_features(x)
            pattern = self.extract_pattern_features(x)
            
            # Combine and return np array
            combined = np.concatenate([temporal, frequency, pattern])
            all_features.append(combined)
        
        return np.array(all_features)

def test_multiclass():
    """Load data and test"""
    
    # Load data for all gestures (5 currently)
    gesture_files = [
        'cleaned_control_log.csv',
        'cleaned_wave_log.csv',
        'cleaned_clap_log.csv', 
        'cleaned_push_log.csv',
        'cleaned_pull_log.csv'
    ]
    
    gesture_names = ['control', 'wave', 'clap', 'push', 'pull']
    
    X_all = []
    y_all = []
    
    for gesture_idx, filename in enumerate(gesture_files):
        try:
            X_gesture = load_dataset(filename)
            X_gesture = [np.array(x) for x in X_gesture]
            
            # class labels, takes # of gesture samples and makes vector with label
            y_gesture_labels = np.full(len(X_gesture), gesture_idx)
            
            # add to full list
            X_all.extend(X_gesture)
            y_all.extend(y_gesture_labels)
            
            print(f"Loaded {len(X_gesture)} samples for {gesture_names[gesture_idx]}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return 0.0
    
    y_all = np.array(y_all)
    
    
    #Print info about dataset
    print(f"\nTotal samples: {len(X_all)}")
    print(f"Class distribution: {np.bincount(y_all)}")
    for i, count in enumerate(np.bincount(y_all)):
        print(f"  {gesture_names[i]}: {count} samples")
    
    # Extract features from loaded files
    print("\nExtracting features...")
    extractor = GestureFeatureExtractor()
    X_features = extractor.extract_all_features(X_all)
    
    # Replace any invalid featues with 0
    X_features = np.nan_to_num(X_features)
    
    # Normalize features
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    
    # Range for scaled features 
    print(f"Feature range after scaling: [{X_features_scaled.min():.3f}, {X_features_scaled.max():.3f}]")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Check size
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Runs a Random Forest first, good for a baseline
    print("\n=== RANDOM FOREST ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    
    # Random Forest results
    print(f"Random Forest - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Classification report
    y_pred = rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=gesture_names))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # If RF works well, try neural network
    if test_acc > 0.4:  # this is a pretty low threshold
        print(f"\n=== NEURAL NETWORK ===")
        
        class MultiClassFeatureNet(nn.Module):
            def __init__(self, input_size, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.net(x)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Long for CrossEntropyLoss
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)
        
        # Initialize model
        model = MultiClassFeatureNet(X_features.shape[1], num_classes=5)
        criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Train
        for epoch in range(50):
            model.train()
            running_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Eval every 10th epoch
            if epoch % 10 == 9:
                model.eval()
                with torch.no_grad():
                    train_outputs = model(X_train_tensor)
                    test_outputs = model(X_test_tensor)
                    
                    _, train_preds = torch.max(train_outputs, 1)
                    _, test_preds = torch.max(test_outputs, 1)
                    
                    train_acc = (train_preds == y_train_tensor).float().mean().item()
                    test_acc = (test_preds == y_test_tensor).float().mean().item()
                    
                    print(f"Epoch {epoch+1:2d}: Loss={running_loss/len(train_loader):.4f}, "
                          f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        # Final eval
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, nn_test_preds = torch.max(test_outputs, 1)
            nn_test_acc = (nn_test_preds == y_test_tensor).float().mean().item()
            
        print(f"\nFinal Neural Network Test Accuracy: {nn_test_acc:.4f}")
        print("\nNeural Network Classification Report:")
        print(classification_report(y_test_tensor.numpy(), nn_test_preds.numpy(), target_names=gesture_names))
        
        return max(test_acc, nn_test_acc)
    
    # if RF is not promising NN won't run (think of the power bill)
    else:
        print(f"\nRandom Forest accuracy ({test_acc:.4f}) not great.")

    
    return test_acc

if __name__ == "__main__":
    accuracy = test_multiclass()
    print(f"\nFinal best accuracy: {accuracy:.4f}")
    
    if accuracy < 0.3:
        print("This is just random guessing")
    elif accuracy < 0.6:
        print("Could use some improvement")
    else:
        print("Not too shabby")