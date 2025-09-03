# config/config_validator.py
from typing import Dict, Any, List
import os
from pathlib import Path

class ConfigValidator:
    """Validates a configuration dictionary against a set of rules.

    This class checks for the presence of required sections and fields,
    validates data types and ranges, and ensures that paths exist.
    """

    def __init__(self):
        """Initializes the ConfigValidator and its validation rules."""
        self.validation_rules = self._define_validation_rules()

    def _define_validation_rules(self) -> Dict[str, Any]:
        """Defines the validation rules for the configuration.

        Returns:
            Dict[str, Any]: A dictionary containing the validation rules.
        """
        return {
            'required_sections': [
                'system', 'model', 'training', 'data_sources',
                'inference', 'paths', 'logging'
            ],
            'model_required_fields': {
                'architecture': ['num_layers', 'layer_names'],
                'layers': []  # Will be validated dynamically
            },
            'training_required_fields': {
                'data': ['batch_size'],
                'phases': [],
                'optimizer': ['type']
            },
            'numeric_ranges': {
                'training.data.batch_size': (1, 1024),
                'training.data.train_split': (0.0, 1.0),
                'training.data.val_split': (0.0, 1.0),
                'training.data.test_split': (0.0, 1.0),
                'model.layers.token.dropout': (0.0, 1.0)
            }
        }

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validates the entire configuration and returns a list of errors.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.

        Returns:
            List[str]: A list of error messages. An empty list indicates a
                       valid configuration.
        """
        errors = []

        # Validate required sections exist
        section_errors = self._validate_required_sections(config)
        errors.extend(section_errors)

        # Validate paths exist or can be created
        path_errors = self._validate_paths(config)
        errors.extend(path_errors)

        # Validate model configuration
        if 'model' in config:
            model_errors = self._validate_model_config(config['model'])
            errors.extend(model_errors)

        # Validate training configuration
        if 'training' in config:
            training_errors = self._validate_training_config(config['training'])
            errors.extend(training_errors)

        # Validate data sources
        if 'data_sources' in config:
            data_errors = self._validate_data_sources(config['data_sources'])
            errors.extend(data_errors)

        # Validate numeric ranges
        range_errors = self._validate_numeric_ranges(config)
        errors.extend(range_errors)

        return errors

    def _validate_required_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validates that all required sections are present in the config.

        Args:
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            List[str]: A list of error messages for missing sections.
        """
        errors = []
        required_sections = self.validation_rules['required_sections']

        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required config section: '{section}'")

        return errors

    def _validate_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validates paths in the configuration.

        It checks if directories exist and attempts to create them if they
        do not.

        Args:
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            List[str]: A list of error messages for path-related issues.
        """
        errors = []

        if 'paths' not in config:
            return ["Missing 'paths' section in configuration"]

        paths = config['paths']

        for path_name, path_value in paths.items():
            if path_name.endswith('_dir'):
                path_obj = Path(path_value)
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        print(f"Created directory: {path_value}")
                    except Exception as e:
                        errors.append(f"Cannot create directory '{path_value}': {e}")

        return errors

    def _validate_model_config(self, model_config: Dict[str, Any]) -> List[str]:
        """Validates the model configuration section.

        Args:
            model_config (Dict[str, Any]): The model configuration dictionary.

        Returns:
            List[str]: A list of error messages for model config issues.
        """
        errors = []

        # Check architecture section
        if 'architecture' not in model_config:
            errors.append("Missing 'architecture' section in model config")
            return errors

        architecture = model_config['architecture']

        # Validate required architecture fields
        for field in self.validation_rules['model_required_fields']['architecture']:
            if field not in architecture:
                errors.append(f"Missing required field 'model.architecture.{field}'")

        # Check layer consistency
        if 'layer_names' in architecture and 'layers' in model_config:
            expected_layers = architecture['layer_names']
            configured_layers = list(model_config['layers'].keys())

            missing_layers = set(expected_layers) - set(configured_layers)
            if missing_layers:
                errors.append(f"Missing layer configurations: {missing_layers}")

            extra_layers = set(configured_layers) - set(expected_layers)
            if extra_layers:
                errors.append(f"Unexpected layer configurations: {extra_layers}")

            # Validate each layer configuration
            for layer_name in expected_layers:
                if layer_name in model_config['layers']:
                    layer_config = model_config['layers'][layer_name]
                    layer_errors = self._validate_layer_config(layer_name, layer_config)
                    errors.extend(layer_errors)

        return errors

    def _validate_layer_config(self, layer_name: str, layer_config: Dict[str, Any]) -> List[str]:
        """Validates the configuration for a single layer.

        Args:
            layer_name (str): The name of the layer.
            layer_config (Dict[str, Any]): The configuration for the layer.

        Returns:
            List[str]: A list of error messages for the layer config.
        """
        errors = []

        # Common required fields for all layers
        common_required = ['hidden_dim']

        for field in common_required:
            if field not in layer_config:
                errors.append(f"Layer '{layer_name}' missing required field '{field}'")

        # Layer-specific validation
        if layer_name == 'token':
            token_required = ['embedding_dim', 'vocab_size', 'max_sequence_length']
            for field in token_required:
                if field not in layer_config:
                    errors.append(f"Token layer missing required field '{field}'")

        elif layer_name == 'syntactic':
            syntactic_required = ['num_attention_heads', 'parser_type']
            for field in syntactic_required:
                if field not in layer_config:
                    errors.append(f"Syntactic layer missing required field '{field}'")

        elif layer_name == 'semantic':
            semantic_required = ['entity_embedding_dim', 'entity_types']
            for field in semantic_required:
                if field not in layer_config:
                    errors.append(f"Semantic layer missing required field '{field}'")

        return errors

    def _validate_training_config(self, training_config: Dict[str, Any]) -> List[str]:
        """Validates the training configuration section.

        Args:
            training_config (Dict[str, Any]): The training config dictionary.

        Returns:
            List[str]: A list of error messages for training config issues.
        """
        errors = []

        # Check data splits sum to 1.0 (with tolerance)
        if 'data' in training_config:
            data_config = training_config['data']
            train_split = data_config.get('train_split', 0.8)
            val_split = data_config.get('val_split', 0.1)
            test_split = data_config.get('test_split', 0.1)

            total_split = train_split + val_split + test_split
            if abs(total_split - 1.0) > 0.001:
                errors.append(f"Data splits sum to {total_split}, should sum to 1.0")

        # Validate optimizer configuration
        if 'optimizer' in training_config:
            optimizer_config = training_config['optimizer']
            if 'type' not in optimizer_config:
                errors.append("Missing optimizer type")
            elif optimizer_config['type'] not in ['adam', 'adamw', 'sgd']:
                errors.append(f"Unsupported optimizer type: {optimizer_config['type']}")

        # Validate phases configuration
        if 'phases' in training_config:
            phases = training_config['phases']

            if 'layerwise_pretraining' in phases:
                pretraining = phases['layerwise_pretraining']
                if pretraining.get('enabled', False):
                    if 'epochs_per_layer' not in pretraining:
                        errors.append("Layerwise pretraining enabled but missing 'epochs_per_layer'")

        return errors

    def _validate_data_sources(self, data_sources: Dict[str, Any]) -> List[str]:
        """Validates the data source configurations.

        Args:
            data_sources (Dict[str, Any]): The data sources dictionary.

        Returns:
            List[str]: A list of error messages for data source issues.
        """
        errors = []

        enabled_sources = []
        for source_name, source_config in data_sources.items():
            if isinstance(source_config, dict) and source_config.get('enabled', False):
                enabled_sources.append(source_name)

                # Check if path exists (only for enabled sources)
                if 'path' in source_config:
                    path = source_config['path']
                    if not os.path.exists(path):
                        # For training, this might be a warning rather than error
                        errors.append(f"Warning: Data source '{source_name}' path does not exist: {path}")

        # Check that at least one data source is enabled for training
        if not enabled_sources:
            errors.append("No data sources are enabled")

        return errors

    def _validate_numeric_ranges(self, config: Dict[str, Any]) -> List[str]:
        """Validates that numeric values are within expected ranges.

        Args:
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            List[str]: A list of error messages for out-of-range values.
        """
        errors = []

        for config_path, (min_val, max_val) in self.validation_rules['numeric_ranges'].items():
            value = self._get_nested_value(config, config_path)

            if value is not None:
                if not isinstance(value, (int, float)):
                    errors.append(f"'{config_path}' should be numeric, got {type(value)}")
                elif not (min_val <= value <= max_val):
                    errors.append(f"'{config_path}' = {value} is outside valid range [{min_val}, {max_val}]")

        return errors

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Retrieves a value from a nested dictionary using a dot-separated path.

        Args:
            config (Dict[str, Any]): The dictionary to search.
            path (str): The dot-separated path to the value.

        Returns:
            Any: The value if found, otherwise None.
        """
        keys = path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

# Convenience function for validation
def validate_config(config: Dict[str, Any]) -> List[str]:
    """A convenience function to validate a configuration.

    This function creates a new instance of the ConfigValidator and runs the
    validation.

    Args:
        config (Dict[str, Any]): The configuration dictionary to validate.

    Returns:
        List[str]: A list of error messages.
    """
    validator = ConfigValidator()
    return validator.validate(config)
