import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('.').resolve()))

from views.comparison import (
    load_model_comparison_data,
    ALL_METRICS,
    MODEL_ORDER,
    LIFECYCLE_STAGES,
    SPLIT_STAGES,
    _load_from_final_comparison,
    _load_from_model_artifacts,
    _load_split_data,
    _calculate_majority_baseline,
)

print("--- Test 1: Full load_model_comparison_data() execution ---")
data = load_model_comparison_data()
assert "error" not in data, f"Unexpected error: {data.get('error')}"
assert data["model_order"] == MODEL_ORDER
assert data["lifecycle_stages"] == LIFECYCLE_STAGES
assert data["split_stages"] == SPLIT_STAGES
assert "majority_acc" in data and isinstance(data["majority_acc"], float)
assert "majority_f1" in data and isinstance(data["majority_f1"], float)
assert "lifecycle_matrix_df" in data and not data["lifecycle_matrix_df"].empty

print(f"Majority Acc: {data['majority_acc']}")
print(f"Majority F1: {data['majority_f1']}")
print(f"Lifecycle Matrix DF Rows: {len(data['lifecycle_matrix_df'])}")

for m in ALL_METRICS:
    assert m in data["lifecycle_data"], f"Missing metric {m} in lifecycle_data"
    for model in MODEL_ORDER:
        assert model in data["lifecycle_data"][m], f"Missing {model} in {m}"
        scores = data["lifecycle_data"][m][model]
        assert len(scores) == 3, f"Expected 3 scores for {model} in {m}, got {len(scores)}"
        assert all(isinstance(s, (float, int)) for s in scores)
print("All lifecycle_data metrics validated successfully!")

for m in ["Accuracy", "Macro F1", "ROC-AUC"]:
    assert m in data["split_data"], f"Missing {m} in split_data"
    for model in MODEL_ORDER:
        assert model in data["split_data"][m], f"Missing {model} in split_data[{m}]"
        scores = data["split_data"][m][model]
        assert len(scores) == 3, f"Expected 3 scores for {model} in split_data[{m}]"
print("All split_data metrics validated successfully!")

print("\n--- Test 2: Test fallback loader directly ---")
artifacts_dir = Path('.').resolve() / "artifacts"
fallback_data = _load_from_model_artifacts(artifacts_dir, MODEL_ORDER, ALL_METRICS)
assert fallback_data is not None, "Fallback loader returned None"
for m in ALL_METRICS:
    for model in MODEL_ORDER:
        assert model in fallback_data[m]
print("Fallback loader directly validated successfully!")

print("\n--- Test 3: Test missing artifact directory error handling ---")
fake_dir = Path('.').resolve() / "non_existent_artifacts_dir"
missing_data = _load_from_final_comparison(fake_dir, MODEL_ORDER, ALL_METRICS)
assert missing_data is None, "Expected None for non-existent dir"
missing_model_data = _load_from_model_artifacts(fake_dir, MODEL_ORDER, ALL_METRICS)
assert missing_model_data is None, "Expected None for non-existent dir in fallback"
print("Missing artifact handling validated successfully!")

print("\nALL INTEGRATION TESTS PASSED!")
