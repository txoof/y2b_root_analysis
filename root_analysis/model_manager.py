# library/model_manager.py
"""Class for managing Keras model training, saving, and loading with metadata.

This manager handles the complete lifecycle of Keras models including training,
saving, loading, and metadata management. It works with any Keras model architecture
including CNNs, U-Nets, ResNets, and custom architectures.

Examples:
    U-Net for image segmentation (Task 5)::
    
        from library.model_manager import KerasModelManager
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Initialize with Task 5 naming convention
        manager = KerasModelManager(
            model_name='smith_123456_unet_model_128px',
            student_name='John Smith',
            student_number='123456',
            patch_size=128,
            architecture='U-Net',
            task='root_segmentation'
        )
        
        # Set model with custom metrics
        manager.set_model(unet_model, custom_objects={'f1': f1})
        
        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                                   restore_best_weights=True)
        manager.train(X_train, y_train, X_val, y_val, 
                     epochs=50, callbacks=[early_stop])
        
        # Save everything
        manager.save('models')
        
        # Get best metrics
        metrics = manager.get_best_metrics()
        print(f"Best F1: {metrics['best_val_f1']:.4f}")
    
    CNN classifier for MNIST::
    
        manager = KerasModelManager(
            model_name='mnist_cnn',
            version='1.0',
            dataset='MNIST',
            input_shape=(28, 28, 1),
            num_classes=10,
            architecture='CNN'
        )
        
        manager.set_model(cnn_model)
        manager.train(X_train, y_train, X_val, y_val, epochs=20)
        manager.save('models')
    
    ResNet50 for plant classification::
    
        manager = KerasModelManager(
            model_name='resnet50_plants',
            version='2.1',
            architecture='ResNet50',
            dataset='PlantCLEF',
            pretrained='ImageNet',
            fine_tuned_layers=10,
            input_size=224
        )
        
        manager.set_model(resnet_model)
        manager.train(train_generator, steps_per_epoch=100,
                     validation_data=val_generator, validation_steps=20,
                     epochs=30)
        manager.save('models')
    
    LSTM for time series prediction::
    
        manager = KerasModelManager(
            model_name='stock_lstm',
            version='1.0',
            architecture='LSTM',
            sequence_length=60,
            features=5,
            prediction_horizon=1
        )
        
        manager.set_model(lstm_model)
        manager.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        manager.save('models')
    
    Loading a saved model::
    
        # Initialize manager with same naming
        manager = KerasModelManager(
            model_name='smith_123456_unet_model_128px'
        )
        
        # Set custom objects before loading
        manager.set_model(None, custom_objects={'f1': f1})
        
        # Load model and history
        model, history = manager.load('models/smith_123456_unet_model_128px.h5')
        
        # Use loaded model
        predictions = model.predict(X_test)

Attributes:
    model_name (str): Base name for the model files.
    version (str): Optional version string or number.
    metadata (dict): Additional metadata stored with the model.
    model (keras.Model): The Keras model being managed.
    history (keras.callbacks.History): Training history from model.fit().
    custom_objects (dict): Custom objects needed for model loading.
    base_name (str): Generated base filename.
    model_path (str): Path to saved model file.
    history_path (str): Path to saved history JSON.
    summary_path (str): Path to saved summary text file.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tensorflow import keras


class KerasModelManager:
    """Manages Keras model lifecycle: training, saving, loading, and metadata."""
    
    def __init__(self, model_name, version=None, notes=None, **metadata):
        """
        Initialize the model manager.
        
        Args:
            model_name: Base name for the model files. For Task 5, use format:
                       'studentname_studentnumber_modeltype_patchsizepx'
            version: Optional version string or number (e.g., '1.0', 'v2', 2).
            notes: Optional initial note/hypothesis for this model.
            **metadata: Additional metadata to store with the model. Common examples:
                       - student_name (str): Student name for academic projects
                       - student_number (str): Student number for academic projects
                       - architecture (str): Model architecture (e.g., 'U-Net', 'ResNet50')
                       - patch_size (int): Patch size for segmentation models
                       - input_shape (tuple): Input shape for the model
                       - num_classes (int): Number of output classes
                       - dataset (str): Dataset name
                       - pretrained (str): Pretrained weights source
                       - task (str): Task description
        
        Examples:
            >>> # U-Net for segmentation with notes
            >>> manager = KerasModelManager(
            ...     'smith_123456_unet_model_128px',
            ...     notes="Hypothesis: 128px patches will achieve F1 > 0.90",
            ...     patch_size=128,
            ...     architecture='U-Net'
            ... )
            
            >>> # Versioned CNN classifier
            >>> manager = KerasModelManager(
            ...     'mnist_cnn',
            ...     version='1.0',
            ...     dataset='MNIST',
            ...     input_shape=(28, 28, 1)
            ... )
        """
        self.model_name = model_name
        self.version = version
        self.metadata = metadata
        self.model = None
        self.history = None
        self.custom_objects = {}
        self._callbacks = None
        self._notes = []
        
        # Add initial note if provided
        if notes:
            self._add_note(notes)
        
        # Generate filenames
        version_str = f"_v{version}" if version else ""
        self.base_name = f"{model_name}{version_str}"
        self.model_path = f"{self.base_name}.h5"
        self.history_path = f"{self.base_name}_history.json"
        self.summary_path = f"{self.base_name}_summary.txt"
    
    def set_model(self, model, custom_objects=None):
        """
        Set the model to be managed.
        
        Args:
            model: Compiled Keras model, or None if only loading.
            custom_objects: Dict of custom objects needed for model loading
                           (e.g., {'f1': f1_function, 'dice_loss': dice_loss_fn}).
        
        Examples:
            >>> # With custom metric
            >>> manager.set_model(unet_model, custom_objects={'f1': f1})
            
            >>> # Standard model, no custom objects
            >>> manager.set_model(resnet_model)
            
            >>> # Preparing to load (no model yet)
            >>> manager.set_model(None, custom_objects={'f1': f1})
        """
        self.model = model
        if custom_objects:
            self.custom_objects = custom_objects
    
    def _add_note(self, note):
        """Internal method to add a timestamped note."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._notes.append({
            'timestamp': timestamp,
            'note': note
        })
    
    @property
    def notes(self):
        """Get all notes as a formatted string."""
        if not self._notes:
            return ""
        return "\n".join([f"[{n['timestamp']}] {n['note']}" for n in self._notes])
    
    @notes.setter
    def notes(self, note):
        """Append a new note with automatic timestamp."""
        self._add_note(note)
    
    def set_notes(self, note, replace=False):
        """
        Set notes with option to replace all existing notes.
        
        Args:
            note: The note text to add or set.
            replace: If True, replaces all existing notes. If False, appends.
        
        Examples:
            >>> # Append (default)
            >>> manager.set_notes("Added dropout layers")
            
            >>> # Replace all notes
            >>> manager.set_notes("Starting fresh experiment", replace=True)
        """
        if replace:
            self._notes = []
        self._add_note(note)
    
    def clear_notes(self):
        """Clear all notes."""
        self._notes = []
    
    def get_notes_list(self):
        """
        Get notes as a list of dictionaries.
        
        Returns:
            List of dicts with 'timestamp' and 'note' keys.
        """
        return self._notes.copy()
    
    def train(self, X_train, y_train=None, X_val=None, y_val=None, epochs=50, 
            batch_size=16, callbacks=None, verbose=1, **fit_kwargs):
        """
        Train the model and store history.
        
        Args:
            X_train: Training data (numpy array, generator, or dataset).
            y_train: Training labels (numpy array, or None if using generator).
            X_val: Optional validation data (or None if using validation_data kwarg).
            y_val: Optional validation labels (or None if using validation_data kwarg).
            epochs: Maximum number of epochs.
            batch_size: Batch size for training (ignored if using generator).
            callbacks: List of Keras callbacks (e.g., EarlyStopping, ModelCheckpoint).
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).
            **fit_kwargs: Additional arguments passed to model.fit()
                        (e.g., steps_per_epoch, validation_steps, validation_data).
        
        Returns:
            Training history object.
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Store callbacks for later serialization
        self._callbacks = callbacks
        
        # Handle validation_data - check if it's in fit_kwargs first
        validation_data = fit_kwargs.pop('validation_data', None)
        
        # If not in fit_kwargs and we have X_val/y_val, construct it
        if validation_data is None and X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            **fit_kwargs
        )
        
        return self.history
    
    def save(self, output_dir='.'):
        """
        Save model, history, and summary to disk.
        
        Saves three files:
        1. .h5 file - Complete model (architecture, weights, optimizer state)
        2. _history.json - Training metrics for all epochs
        3. _summary.txt - Human-readable summary
        
        Args:
            output_dir: Directory to save files. Created if it doesn't exist.
                       Default is current directory.
        
        Examples:
            >>> # Save to current directory
            >>> manager.save()
            
            >>> # Save to specific directory
            >>> manager.save('trained_models')
            
            >>> # Save to nested path
            >>> manager.save('models/unet/version_1')
        """
        if self.model is None:
            raise ValueError("No model to save. Train or set a model first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_dir / self.model_path
        self.model.save(str(model_file))
        print(f"Model saved: {model_file}")
        
        # Save history if available
        if self.history is not None:
            self._save_history(output_dir)
            self._save_summary(output_dir)
    
    def _serialize_custom_objects(self):
        """
        Serialize information about custom objects for documentation.
        
        Returns:
            List of dictionaries with custom object metadata.
        """
        import inspect
        import hashlib
        
        if not self.custom_objects:
            return []
        
        serialized = []
        
        for name, obj in self.custom_objects.items():
            obj_info = {
                'name': name,
                'type': type(obj).__name__
            }
            
            # Try to get useful information about the object
            try:
                # Get module
                if hasattr(obj, '__module__'):
                    obj_info['module'] = obj.__module__
                
                # Get docstring
                if hasattr(obj, '__doc__') and obj.__doc__:
                    obj_info['docstring'] = obj.__doc__.strip()
                
                # For functions, get signature
                if callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        obj_info['signature'] = str(sig)
                    except (ValueError, TypeError):
                        pass
                    
                    # Get source code hash (for version tracking)
                    try:
                        source = inspect.getsource(obj)
                        source_hash = hashlib.md5(source.encode()).hexdigest()
                        obj_info['source_hash'] = source_hash
                    except (OSError, TypeError):
                        pass
                
                serialized.append(obj_info)
                
            except Exception as e:
                # If we can't serialize, at least save the name and type
                obj_info['error'] = str(e)
                serialized.append(obj_info)
        
        return serialized
    
    def _serialize_callbacks(self, callbacks):
        """
        Serialize callback configurations.
        
        Args:
            callbacks: List of Keras callbacks or None.
        
        Returns:
            List of serialized callback dictionaries.
        """
        import warnings
        
        if callbacks is None:
            return []
        
        serialized = []
        
        for callback in callbacks:
            callback_info = {
                'class': callback.__class__.__name__,
                'module': callback.__class__.__module__
            }
            
            # Try to extract configuration
            try:
                # Get callback config if available
                if hasattr(callback, 'get_config'):
                    callback_info['config'] = callback.get_config()
                else:
                    # Manually extract common attributes
                    config = {}
                    
                    # EarlyStopping attributes
                    if hasattr(callback, 'monitor'):
                        config['monitor'] = callback.monitor
                    if hasattr(callback, 'patience'):
                        config['patience'] = callback.patience
                    if hasattr(callback, 'restore_best_weights'):
                        config['restore_best_weights'] = callback.restore_best_weights
                    if hasattr(callback, 'mode'):
                        config['mode'] = callback.mode
                    if hasattr(callback, 'min_delta'):
                        config['min_delta'] = callback.min_delta
                    
                    # ReduceLROnPlateau attributes
                    if hasattr(callback, 'factor'):
                        config['factor'] = callback.factor
                    if hasattr(callback, 'cooldown'):
                        config['cooldown'] = callback.cooldown
                    if hasattr(callback, 'min_lr'):
                        config['min_lr'] = callback.min_lr
                    
                    # ModelCheckpoint attributes
                    if hasattr(callback, 'filepath'):
                        config['filepath'] = callback.filepath
                    if hasattr(callback, 'save_best_only'):
                        config['save_best_only'] = callback.save_best_only
                    if hasattr(callback, 'save_weights_only'):
                        config['save_weights_only'] = callback.save_weights_only
                    
                    # LearningRateScheduler
                    if hasattr(callback, 'schedule'):
                        warnings.warn(
                            f"Callback {callback.__class__.__name__} has a 'schedule' "
                            f"function that cannot be serialized. Saving class name only.",
                            UserWarning
                        )
                        config['schedule'] = '<function>'
                    
                    if config:
                        callback_info['config'] = config
                    else:
                        warnings.warn(
                            f"Could not extract configuration from callback "
                            f"{callback.__class__.__name__}. Saving class name only.",
                            UserWarning
                        )
                
                # Check if config is JSON serializable
                json.dumps(callback_info)
                serialized.append(callback_info)
                
            except (TypeError, ValueError) as e:
                warnings.warn(
                    f"Callback {callback.__class__.__name__} could not be fully "
                    f"serialized: {str(e)}. Saving class name only.",
                    UserWarning
                )
                serialized.append({
                    'class': callback.__class__.__name__,
                    'module': callback.__class__.__module__,
                    'error': str(e)
                })
        
        return serialized
    
    def _save_history(self, output_dir):
        """Save training history to JSON."""
        history_dict = {}
        
        for key, values in self.history.history.items():
            # Convert to list and ensure JSON serializable
            if isinstance(values, np.ndarray):
                history_dict[key] = values.tolist()
            elif isinstance(values, list):
                history_dict[key] = [float(x) for x in values]
            else:
                history_dict[key] = values
        
        # Serialize callbacks
        callbacks_info = self._serialize_callbacks(self._callbacks)
        
        # Serialize custom objects
        custom_objects_info = self._serialize_custom_objects()
        
        # Add metadata
        history_dict['metadata'] = {
            'model_name': self.model_name,
            'version': self.version,
            'epochs_trained': len(self.history.history['loss']),
            'saved_at': datetime.now().isoformat(),
            'callbacks': callbacks_info,
            'custom_objects': custom_objects_info,
            'notes': self._notes,
            **self.metadata
        }
        
        history_file = output_dir / self.history_path
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=4)
        print(f"History saved: {history_file}")
    
    def _save_summary(self, output_dir):
        """Save human-readable summary."""
        summary_file = output_dir / self.summary_path
        
        with open(summary_file, 'w') as f:
            f.write(f"{self.model_name} Model Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Write metadata
            if self.version:
                f.write(f"Version: {self.version}\n")
            for key, value in self.metadata.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"Model file: {self.model_path}\n")
            f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write notes
            if self._notes:
                f.write("Notes / Hypothesis:\n")
                f.write("-" * 60 + "\n")
                for note_entry in self._notes:
                    f.write(f"[{note_entry['timestamp']}] {note_entry['note']}\n")
                f.write("\n")
            
            # Write custom objects information
            if self.custom_objects:
                f.write("Custom Objects:\n")
                f.write("-" * 60 + "\n")
                custom_objs_info = self._serialize_custom_objects()
                for obj_info in custom_objs_info:
                    f.write(f"  {obj_info['name']} ({obj_info['type']}):\n")
                    if 'module' in obj_info:
                        f.write(f"    module: {obj_info['module']}\n")
                    if 'signature' in obj_info:
                        f.write(f"    signature: {obj_info['signature']}\n")
                    if 'source_hash' in obj_info:
                        f.write(f"    source_hash: {obj_info['source_hash']}\n")
                    if 'docstring' in obj_info:
                        # First line of docstring only
                        first_line = obj_info['docstring'].split('\n')[0]
                        f.write(f"    docstring: {first_line}\n")
                    f.write("\n")
            
            # Write callback information
            if self._callbacks:
                f.write("Callbacks Used:\n")
                f.write("-" * 60 + "\n")
                callbacks_info = self._serialize_callbacks(self._callbacks)
                for cb_info in callbacks_info:
                    f.write(f"  {cb_info['class']}:\n")
                    if 'config' in cb_info:
                        for key, value in cb_info['config'].items():
                            f.write(f"    {key}: {value}\n")
                    elif 'error' in cb_info:
                        f.write(f"    (serialization error: {cb_info['error']})\n")
                    f.write("\n")
            
            if self.history:
                history = self.history.history
                f.write("Training Results\n")
                f.write("-" * 60 + "\n")
                f.write(f"Epochs trained: {len(history['loss'])}\n\n")
                
                # Final metrics
                f.write("Final Metrics:\n")
                for key in sorted(history.keys()):
                    if not key.startswith('val_'):
                        val_key = f'val_{key}'
                        train_val = history[key][-1]
                        f.write(f"  {key}: {train_val:.4f}")
                        if val_key in history:
                            val_val = history[val_key][-1]
                            f.write(f" | {val_key}: {val_val:.4f}")
                        f.write("\n")
                
                # Best metrics (find all val_ metrics and report best)
                f.write("\nBest Validation Metrics:\n")
                for key in sorted(history.keys()):
                    if key.startswith('val_'):
                        # Determine if higher or lower is better
                        if 'loss' in key or 'error' in key:
                            best_idx = np.argmin(history[key])
                            best_val = history[key][best_idx]
                            f.write(f"  {key}: {best_val:.4f} (epoch {best_idx + 1}, min)\n")
                        else:
                            best_idx = np.argmax(history[key])
                            best_val = history[key][best_idx]
                            f.write(f"  {key}: {best_val:.4f} (epoch {best_idx + 1}, max)\n")
        
        print(f"Summary saved: {summary_file}")
    
    def load(self, model_path=None, load_history=True):
        """
        Load a saved model and optionally its history.
        
        Args:
            model_path: Path to model file. If None, uses default naming convention.
            load_history: Whether to load training history JSON. Default is True.
        
        Returns:
            Tuple of (loaded_model, history_dict or None).
        
        Examples:
            >>> # Load with default path
            >>> manager = KerasModelManager('smith_123456_unet_model_128px')
            >>> manager.set_model(None, custom_objects={'f1': f1})
            >>> model, history = manager.load()
            
            >>> # Load from specific path
            >>> model, history = manager.load('models/backup/model.h5')
            
            >>> # Load model only, skip history
            >>> model, _ = manager.load(load_history=False)
        """
        if model_path is None:
            model_path = self.model_path
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with custom objects
        self.model = keras.models.load_model(
            str(model_path),
            custom_objects=self.custom_objects
        )
        print(f"Model loaded: {model_path}")
        
        loaded_history = None
        
        # Load history if available
        if load_history:
            history_path = model_path.parent / self.history_path
            if history_path.exists():
                with open(history_path, 'r') as f:
                    loaded_history = json.load(f)
                    print(f"History loaded: {history_path}")
                    
                    # Print summary
                    if 'metadata' in loaded_history:
                        meta = loaded_history['metadata']
                        print(f"Trained for {meta.get('epochs_trained', 'unknown')} epochs")
                        
                        # Print notes if available
                        if 'notes' in meta and meta['notes']:
                            print("\nNotes:")
                            for note_entry in meta['notes']:
                                print(f"  [{note_entry['timestamp']}] {note_entry['note']}")
                        
                        # Print custom objects if available
                        if 'custom_objects' in meta and meta['custom_objects']:
                            print("\nCustom objects used:")
                            for obj_info in meta['custom_objects']:
                                print(f"  - {obj_info['name']} ({obj_info['type']})")
                        
                        # Print other metadata
                        for key, val in meta.items():
                            if key not in ['model_name', 'version', 'epochs_trained', 'saved_at', 'callbacks', 'custom_objects', 'notes']:
                                print(f"{key}: {val}")
                    
                    # Print best validation metrics
                    for key in loaded_history.keys():
                        if key.startswith('val_') and key != 'val_loss':
                            best_val = max(loaded_history[key])
                            print(f"Best {key}: {best_val:.4f}")
        
        return self.model, loaded_history
    
    def get_best_metrics(self, metric_preferences=None):
        """
        Get best metrics from training history.
        
        Args:
            metric_preferences: Dict mapping metric names to 'min' or 'max'.
                               If None, uses 'min' for loss/error metrics,
                               'max' for all others.
        
        Returns:
            Dictionary with best metrics, their epochs, and final values.
            Keys include 'best_{metric}', 'best_{metric}_epoch', 'final_{metric}'.
        
        Examples:
            >>> metrics = manager.get_best_metrics()
            >>> print(f"Best F1: {metrics['best_val_f1']:.4f}")
            >>> print(f"Achieved at epoch: {metrics['best_val_f1_epoch']}")
            
            >>> # Custom metric preferences
            >>> metrics = manager.get_best_metrics(
            ...     metric_preferences={'val_custom': 'min'}
            ... )
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        history = self.history.history
        metrics = {}
        
        # Default preferences
        if metric_preferences is None:
            metric_preferences = {}
        
        for key in history.keys():
            if key.startswith('val_'):
                # Determine optimization direction
                if key in metric_preferences:
                    direction = metric_preferences[key]
                elif 'loss' in key or 'error' in key:
                    direction = 'min'
                else:
                    direction = 'max'
                
                # Find best value
                if direction == 'min':
                    best_idx = np.argmin(history[key])
                else:
                    best_idx = np.argmax(history[key])
                
                metrics[f'best_{key}'] = history[key][best_idx]
                metrics[f'best_{key}_epoch'] = best_idx + 1
                metrics[f'final_{key}'] = history[key][-1]
        
        return metrics
    
    def plot_history(self, history_dict=None, metrics=None, figsize=(14, 5)):
        """
        Plot training history.
        
        Args:
            history_dict: History dict from load(). If None, uses self.history.
            metrics: List of metrics to plot. If None, plots all metrics.
            figsize: Figure size tuple.
        
        Examples:
            >>> # After training
            >>> manager.plot_history()
            
            >>> # After loading
            >>> model, history = manager.load()
            >>> manager.plot_history(history)
            
            >>> # Plot specific metrics
            >>> manager.plot_history(metrics=['loss', 'f1'])
        """
        import matplotlib.pyplot as plt
        
        if history_dict is None:
            if self.history is None:
                raise ValueError("No history available. Train or load a model first.")
            history_dict = self.history.history
        
        # Filter out metadata
        history_dict = {k: v for k, v in history_dict.items() if k != 'metadata'}
        
        # Determine metrics to plot
        if metrics is None:
            # Plot all non-validation metrics
            metrics = [k for k in history_dict.keys() if not k.startswith('val_')]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            val_metric = f'val_{metric}'
            
            ax.plot(history_dict[metric], label=f'Training {metric}')
            if val_metric in history_dict:
                ax.plot(history_dict[val_metric], label=f'Validation {metric}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
