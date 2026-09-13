import numpy as np

from src.rolling_24_comparison import recursive_tabular_predictions


class IncrementLastHistory:
    def predict(self, features):
        return features[:, -1] + 1.0


def test_recursive_tabular_predictions_only_reuse_own_predictions():
    summaries = np.zeros((40, 3, 4), dtype=np.float32)
    histories = np.zeros((2, 24), dtype=np.float32)
    origins = np.asarray([24, 25])
    predicted = recursive_tabular_predictions(
        IncrementLastHistory(), summaries, histories, origins, output_steps=4
    )
    np.testing.assert_array_equal(predicted, [[1, 2, 3, 4], [1, 2, 3, 4]])
