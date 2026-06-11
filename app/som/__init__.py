"""
SOM module — standalone unit for Self-Organizing Map training and analysis.

Public API (the only contract other modules — EA, LSTM, UI, tools — should use):

    from som import KohonenSOM, preprocess_data, PreprocessResult
    from som import persistence

Layers:
    som.som            core algorithm + map-quality metrics (pure compute)
    som.preprocess     input stage: validate, classify, encode, normalize (pure)
    som.persistence    all disk writes for a results directory
    som.analysis       post-training BMU assignment, extremes, pie data
    som.visualization  map rendering from stored artifacts (weights + map_type)
    som.graphs         training-history plots
    som.run            CLI orchestrator; the only place with outward imports

Heavy imports (matplotlib in visualization/graphs) are intentionally NOT
re-exported here — import those submodules explicitly when needed.
"""
from som.som import KohonenSOM
from som.preprocess import preprocess_data, validate_input_data, PreprocessResult
from som import persistence

__all__ = [
    'KohonenSOM',
    'preprocess_data',
    'validate_input_data',
    'PreprocessResult',
    'persistence',
]
