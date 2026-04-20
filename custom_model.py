"""
Custom Logistic Regression — Built From Scratch (NumPy Only)
=============================================================
Demonstrates foundational ML knowledge without relying on scikit-learn.

This implementation includes:
  - Sigmoid activation function
  - Binary cross-entropy (log-loss) cost function
  - Gradient descent optimisation with learning-rate scheduling
  - L2 regularisation (Ridge penalty)
  - fit() / predict() / predict_proba() API compatible with scikit-learn

Author: Sourav
"""

import numpy as np


class CustomLogisticRegression:
    """
    A pure-NumPy implementation of binary Logistic Regression
    trained via batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Initial step size for gradient descent.
    n_iterations : int
        Maximum number of gradient descent updates.
    regularization : float
        L2 (Ridge) penalty strength. Set to 0 for no regularisation.
    threshold : float
        Decision boundary for converting probabilities to class labels.
    verbose : bool
        If True, print the loss every 100 iterations.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 3000,
        regularization: float = 0.01,
        threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.threshold = threshold
        self.verbose = verbose

        # Parameters learned during fit()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ──────────────────────────────────────────
    #  Core mathematical building blocks
    # ──────────────────────────────────────────

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Clip to avoid overflow in exp()
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Binary cross-entropy loss with L2 regularisation.

        L = -1/m * Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]  +  λ/(2m) * ||w||²
        """
        m = len(y)
        epsilon = 1e-15  # prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        cross_entropy = -np.mean(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        )
        l2_penalty = (self.regularization / (2 * m)) * np.sum(self.weights ** 2)

        return cross_entropy + l2_penalty

    # ──────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomLogisticRegression":
        """
        Train the logistic regression model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (m, n)
            Feature matrix (already scaled/encoded).
        y : np.ndarray of shape (m,)
            Binary target labels (0 or 1).

        Returns
        -------
        self
        """
        m, n = X.shape

        # Initialise weights to small random values (Xavier-style)
        self.weights = np.random.randn(n) * np.sqrt(2.0 / n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(1, self.n_iterations + 1):
            # Forward pass
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Compute gradients
            error = y_hat - y
            dw = (1 / m) * (X.T @ error) + (self.regularization / m) * self.weights
            db = (1 / m) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._compute_loss(y, y_hat)
            self.loss_history.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"    Iteration {i:>5d}/{self.n_iterations}  |  Loss: {loss:.6f}")

        return self

    # ──────────────────────────────────────────
    #  Prediction (scikit-learn compatible API)
    # ──────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities as a (m, 2) array.

        Column 0 = P(class 0), Column 1 = P(class 1).
        Mirrors sklearn's predict_proba interface.
        """
        z = X @ self.weights + self.bias
        prob_positive = self._sigmoid(z)
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    # ──────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────

    def get_params(self) -> dict:
        """Return the learned weights and bias."""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "n_features": len(self.weights) if self.weights is not None else 0,
        }

    def __repr__(self) -> str:
        return (
            f"CustomLogisticRegression("
            f"lr={self.learning_rate}, "
            f"iters={self.n_iterations}, "
            f"reg={self.regularization})"
        )
