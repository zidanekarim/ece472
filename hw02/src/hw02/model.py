from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from sklearn.base import BaseEstimator, ClassifierMixin

@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float


class SK_NNXClassifier(BaseEstimator, ClassifierMixin):
    """Allows a Flax NNX model to be used with scikit-learn. DecisionBoundaryDisplay 
    builds a 2d grid of points and calls the model to get predictions for each point. This wrapper is necessary because the model is not a scikit-learn model and does not have a predict method. 
    """ 
    def __init__(self, model):
        self.model = model
         # 0, 1 because its either red or blue. this is used by DecisionBoundaryDisplay to determine the classes of the model.
    
    def predict_proba(self, X):
        """Returns the predicted class probabilities for the input data."""
        X = jnp.asarray(X)
        logits = self.model(X)
        probs = jax.nn.softmax(logits, axis=-1)
        return np.array(probs) # convert back to numpy array for sklearn 
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1) # returns the index of the max probability

    def fit(self, X,y=None):
        self.classes_ = np.array([0,1] )
        return self # had to do this because sklearn fails otherwise

class MLP(nnx.Module):
    

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, hidden_layer_width: int, hidden_activation=nnx.gelu, output_activation=nnx.identity, *, rngs: nnx.Rngs):
        layers = []
        self.hidden_activation = hidden_activation
        self.output_activation=output_activation
        
        layers.append(NNXLinearModel(rngs, num_inputs, hidden_layer_width)) # first layer hidden
        for i in range(num_hidden_layers-1):
            layers.append(NNXLinearModel(rngs, hidden_layer_width, hidden_layer_width))
        
        self.hidden_layers = nnx.List(layers)
        self.out_layer = NNXLinearModel(rngs, hidden_layer_width, num_outputs) 

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = self.hidden_activation(layer(x)) # at each layer, compute layer(x) = xW + b. then the hidden actrivation is applying the sigmoid or gelu. this then feeds as the next input (sort of recursive calls?)
        return self.output_activation(self.out_layer(x)) # take final-1 input, feed it to out_layer, and feed output_activation function to that



class SwiGLU(nnx.Module):
    def __init__(self,  num_inputs: int, num_outputs: int,*, rngs:nnx.Rngs):
        self.W1 = NNXLinearModel(rngs, num_inputs, num_outputs)
        self.W2 = NNXLinearModel(rngs, num_inputs, num_outputs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.silu(self.W1(x))
        return gate * self.W2(x) # element wise multiplication (gate symbol). see Shazeer section 2 equation 5 


class SwiGLUWrapper(nnx.Module):
    def __init__(self, dimension: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dimension, rngs=rngs) # without normalization, SwiGLU loss is extremely high (50+)
        self.swiglu = SwiGLU(dimension, dimension, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.swiglu(self.norm(x)) # h = x + SwiGLU(X), see https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/ for where I saw resnet formula


class SwiGLUMLP(nnx.Module):

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, hidden_layer_width: int, output_activation=nnx.identity, *, rngs: nnx.Rngs):
        self.output_activation=output_activation
        self.in_layer  = NNXLinearModel(rngs=rngs, in_features=num_inputs, out_features=hidden_layer_width)
        layers = []
        for i in range(num_hidden_layers):
            layers.append(SwiGLUWrapper(hidden_layer_width, rngs=rngs)) 
        self.layers = nnx.List(layers)
        self.out_layer = NNXLinearModel( rngs=rngs, in_features=hidden_layer_width, out_features=num_outputs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.in_layer(x)
        for layer in self.layers:
            x = layer(x) # x = X + SiLU(w1 x) * (w2 x), residual connection for each layer. same stuff as MLP just formula change, and instead of hidden activation we just SiLU
        return self.output_activation(self.out_layer(x))
        

class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model, with multi-dimension support added to solve MLP"""

    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
        key = rngs.params()
        self.in_features = in_features # bc of MLP
        self.out_features = out_features
        std = jnp.sqrt(1.0 / in_features) # prevents vanishing gradients 
        self.w = nnx.Param(jax.random.normal(key, (in_features, out_features)) * std)
        self.b = nnx.Param(jnp.zeros((1, out_features))) # broadcasts for output

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return x @ self.w.value + self.b.value # squeeze gone to maintain hidden layers 

    @property
    def model(self) -> LinearModel:
        """Returns the underlying simple linear model."""
        return LinearModel(
            weights=np.array(self.w.value).reshape([self.num_features]),
            bias=float(np.array(self.b.value).squeeze()),
        )
