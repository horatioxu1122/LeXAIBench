import importlib.util
from importlib.metadata import PackageNotFoundError, version


OPTIONAL_MODULES = {
    "numpy": "Numerical arrays",
    "pandas": "Tabular dataframes",
    "sklearn": "Classical ML models",
    "xgboost": "Gradient-boosted trees",
    "interpret": "Explainable Boosting Machines",
    "shap": "SHAP explainers",
    "lime": "LIME explainers",
}


EMPIRICAL_GROUPS = {
    "demo_mode": [],
    "first_real_slice": ["numpy", "pandas", "sklearn", "shap"],
    "full_empirical_core": ["numpy", "pandas", "sklearn", "xgboost", "shap", "lime"],
    "full_empirical_plus_ebm": [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "interpret",
        "shap",
        "lime",
    ],
}


def detect_capabilities() -> dict[str, dict[str, str | bool | None]]:
    capabilities: dict[str, dict[str, str | bool | None]] = {}
    for module_name, description in OPTIONAL_MODULES.items():
        available = importlib.util.find_spec(module_name) is not None
        capabilities[module_name] = {
            "available": available,
            "description": description,
            "version": _module_version(module_name) if available else None,
        }
    return capabilities


def capability_matrix() -> dict[str, bool]:
    return {
        module_name: bool(item["available"])
        for module_name, item in detect_capabilities().items()
    }


def empirical_readiness() -> dict[str, dict[str, bool | list[str]]]:
    matrix = capability_matrix()
    report: dict[str, dict[str, bool | list[str]]] = {}
    for group_name, modules in EMPIRICAL_GROUPS.items():
        missing = [module_name for module_name in modules if not matrix.get(module_name, False)]
        report[group_name] = {
            "ready": not missing,
            "missing_modules": missing,
        }
    return report


def _module_version(module_name: str) -> str | None:
    distribution_name = {
        "sklearn": "scikit-learn",
    }.get(module_name, module_name)
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None
