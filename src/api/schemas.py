"""
src/api/schemas.py

Pydantic response schemas for the FastAPI backend.
All schemas use strict types — no numpy types allowed in JSON responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: int


class PricePoint(BaseModel):
    date: str
    spx: float
    gold: float
    btc: float


class RegimePoint(BaseModel):
    date: str
    regime_label: str
    crisis_flag: int
    btc_garch: Optional[float] = None


class MetricsOverall(BaseModel):
    accuracy: float
    macro_f1: float
    weighted_f1: float


class MetricsPerClass(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricsCrisis(BaseModel):
    pr_auc: float
    roc_auc: float
    optimal_threshold: float


class ModelMetrics(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    version: str
    timestamp: str
    overall: MetricsOverall
    per_class: Dict[str, MetricsPerClass]
    crisis_specific: MetricsCrisis
    black_swan: Dict[str, Any]
    confusion_matrix: List[List[int]]


class BenchmarkComparison(BaseModel):
    generated_at: str
    best_model_by_crisis_pr_auc: str
    models: List[str]
    comparison: Dict[str, Dict[str, float]]


class PredictionPoint(BaseModel):
    date: str
    true_regime: int
    pred_regime: int
    prob_crisis: float
    prob_normal: float
    prob_highvol: float


class ConfusionMatrixResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    matrix: List[List[int]]
    class_names: List[str]
    model_name: str


class EntanglementPoint(BaseModel):
    date: str
    entropy: float
    true_regime: int


class LeadLagPoint(BaseModel):
    date: str
    vqh_prob_crisis: float
    lstm_prob_crisis: Optional[float] = None


class LatestPrediction(BaseModel):
    date: str
    true_regime: int
    pred_regime: int
    prob_crisis: float
    prob_normal: float
    prob_highvol: float
    match: bool
    entanglement_entropy: Optional[float] = None


class CustomPredictionInput(BaseModel):
    """Input features for custom prediction."""
    btc_return: float
    spx_return: float
    gold_return: float
    btc_volatility: float
    spx_volatility: float
    gold_volatility: float
    btc_spx_corr: float
    btc_gold_corr: float
    spx_gold_corr: float
    vix_level: float


class ModelPrediction(BaseModel):
    """Prediction result from a single model."""
    model_config = {"protected_namespaces": ()}
    model_name: str
    pred_regime: int
    prob_crisis: float
    prob_normal: float
    prob_highvol: float
    confidence: float


class CustomPredictionResponse(BaseModel):
    """Response containing predictions from all models."""
    input_features: CustomPredictionInput
    predictions: List[ModelPrediction]
    best_model: str
