from typing import Callable, List, MutableSequence, Optional, Tuple, Union

import numpy as np

# from tinygrad.tinygrad.jit import TinyJit
from tinygrad.tinygrad.nn.state import get_parameters
from tinygrad.tinygrad.tensor import Tensor
from tinygrad.tinygrad.nn import Linear, optim

DEBUG = 0


def debug_print(*args, **kwargs):
    if DEBUG != 0:
        print(*args, **kwargs)


class set_nograd:
    def __init__(self, val):
        self.val = val
        self.prev = val

    def __enter__(self):
        self.prev, Tensor.no_grad = Tensor.no_grad, self.val

    def __exit__(self, exc_type, exc_value, traceback):
        Tensor.no_grad = self.prev


class NamedLayer:
    def __init__(self, name: Optional[str] = None):
        self.__name__ = name

    def name(self, name: Optional[str] = None):
        if name is not None:
            self.__name__ = name
        return self.__name__

    def __str__(self):
        return f"{self.__class__.__name__}({self.name()})"

    def __repr__(self):
        self.__str__()

    def __hash__(self):
        return hash(self.name())


class Flatten:
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


class Reshape:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.reshape(self.shape)


class GRU(NamedLayer):
    def __init__(
        self,
        input_shape: Union[
            Tuple[int, int], Tuple[int, int, int]
        ],  # (batch_size, steps, features) or (steps, features)
        units: int,
        batch_size: Optional[int] = 1,
        bias: Optional[bool] = False,
        return_sequences: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        # Init the NamedLayer
        super().__init__(name)
        # Asserts for not None and > 0
        for input_size in input_shape:
            assert (
                input_size is not None and input_size > 0
            ), "input_size must be int greater than 0"
        assert units is not None and units > 0, "units must be int greater than 0"

        if len(input_shape) == 3:
            if batch_size is not None:
                assert (
                    input_shape[0] == batch_size
                ), "batch_size must be equal to the first dimension of input_shape"
            batch_size, num_steps, num_features = input_shape
        else:
            num_steps, num_features = input_shape

        # If batch_size is None
        if batch_size is None:
            batch_size = 1

        assert (
            batch_size is not None and batch_size > 0
        ), "batch_size must be int greater than 0"

        self.return_sequences = return_sequences

        expected_w_ih_shape = (num_features, units, 3)
        expected_w_hh_shape = (units, units, 3)
        expected_b_ih_shape = (units, 3)
        expected_b_hh_shape = (units, 3)
        if return_sequences:
            expected_h_prev_shape = (batch_size, num_steps, units)
            expected_w_ho_shape = (units, num_features)
            expected_b_ho_shape = (num_features,)
        else:
            expected_h_prev_shape = (batch_size, units)
            expected_w_ho_shape = (units, num_features)
            expected_b_ho_shape = (num_features,)

        debug_print(f"expected_w_ih_shape: {expected_w_ih_shape}")
        debug_print(f"expected_w_hh_shape: {expected_w_hh_shape}")
        debug_print(f"expected_b_ih_shape: {expected_b_ih_shape}")
        debug_print(f"expected_b_hh_shape: {expected_b_hh_shape}")
        debug_print(f"expected_h_prev_shape: {expected_h_prev_shape}")
        debug_print(f"expected_w_ho_shape: {expected_w_ho_shape}")
        debug_print(f"expected_b_ho_shape: {expected_b_ho_shape}")

        # Initialize weights and biases for the GRU cell
        self.w_ih = Tensor.kaiming_uniform(*expected_w_ih_shape)
        self.w_hh = Tensor.kaiming_uniform(*expected_w_hh_shape)
        self.b_ih = Tensor.zeros(*expected_b_ih_shape) if bias else None
        self.b_hh = Tensor.zeros(*expected_b_hh_shape) if bias else None

        # Initialize the weights and biases for the output layer
        self.w_ho = Tensor.kaiming_uniform(*expected_w_ho_shape)
        self.b_ho = Tensor.zeros(*expected_b_ho_shape) if bias else None

        self.h_prev = Tensor.zeros(*expected_h_prev_shape)

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        # GRU cell computations
        # Check for batch size
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(self.h_prev.shape) == 2:
            self.h_prev = self.h_prev.unsqueeze(0)

        debug_print(f"x.shape: {x.shape}")
        debug_print(f"self.h_prev.shape: {self.h_prev.shape}")

        h_next_seq = []
        y_pred_seq = []

        for batch in range(x.shape[0]):
            batch_x = x[batch]
            batch_h_prev = self.h_prev[batch]
            gi = batch_x.linear(self.w_ih.transpose(), self.b_ih)
            gh = batch_h_prev.linear(self.w_hh.transpose(), self.b_hh)

            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)

            debug_print(f"i_r: {i_r.shape}")
            debug_print(f"i_i: {i_i.shape}")
            debug_print(f"i_n: {i_n.shape}")
            debug_print(f"h_r: {h_r.shape}")
            debug_print(f"h_i: {h_i.shape}")
            debug_print(f"h_n: {h_n.shape}")

            resetgate = (i_r + h_r).sigmoid()
            inputgate = (i_i + h_i).sigmoid()
            newgate = (i_n + (resetgate * h_n)).tanh()

            batch_h_next = (newgate * inputgate) + ((1 - inputgate) * batch_h_prev)

            # Output layer computations
            if self.return_sequences:
                batch_y_pred = batch_h_next.linear(self.w_ho, self.b_ho)
            else:
                batch_y_pred = batch_h_next[-1].linear(self.w_ho, self.b_ho)

            debug_print(f"batch_h_next.shape: {batch_h_next.shape}")
            debug_print(f"batch_y_pred.shape: {batch_y_pred.shape}")

            h_next_seq.append(batch_h_next)
            y_pred_seq.append(batch_y_pred)

        h_next = Tensor.stack(h_next_seq)
        y_pred = Tensor.stack(y_pred_seq)

        debug_print(f"h_next.shape: {h_next.shape}")
        debug_print(f"y_pred.shape: {y_pred.shape}")

        self.h_prev = h_next

        return y_pred


class LSTM(NamedLayer):
    def __init__(
        self,
        input_shape: Union[
            Tuple[int, int], Tuple[int, int, int]
        ],  # (batch_size, steps, features) or (steps, features)
        hidden_size: int,
        num_layers: Optional[int] = 1,
        batch_size: Optional[int] = None,
        bias: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        # Init the NamedLayer
        super().__init__(name)
        # Asserts for not None and > 0
        for input_size in input_shape:
            assert (
                input_size is not None and input_size > 0
            ), "input_size must be int greater than 0"
        assert (
            hidden_size is not None and hidden_size > 0
        ), "hidden_size must be int greater than 0"
        assert (
            num_layers is not None and num_layers > 0
        ), "num_layers must be int greater than 0"

        if len(input_shape) == 3:
            if batch_size is not None:
                assert (
                    input_shape[0] == batch_size
                ), "batch_size must be equal to the first dimension of input_shape"
            batch_size, num_steps, num_features = input_shape
        else:
            num_steps, num_features = input_shape

        # If batch_size is None
        if batch_size is None:
            batch_size = 1

        assert (
            batch_size is not None and batch_size > 0
        ), "batch_size must be int greater than 0"

        expected_w_ih_shape = (hidden_size, num_features, 4)  # Transposed later
        expected_w_hh_shape = (hidden_size, hidden_size, 4)  # Transposed later
        expected_b_ih_shape = (4, hidden_size)
        expected_b_hh_shape = (4, hidden_size)
        expected_h_prev_shape = (num_features, hidden_size)
        expected_w_ho_shape = (hidden_size, num_layers)
        expected_b_ho_shape = (num_layers,)

        # Initialize weights and biases for the GRU cell
        self.w_ih = Tensor.kaiming_uniform(*expected_w_ih_shape)
        self.w_hh = Tensor.kaiming_uniform(*expected_w_hh_shape)
        self.b_ih = Tensor.zeros(*expected_b_ih_shape) if bias else None
        self.b_hh = Tensor.zeros(*expected_b_hh_shape) if bias else None

        # Initialize the weights and biases for the output layer
        self.w_ho = Tensor.kaiming_uniform(*expected_w_ho_shape)
        self.b_ho = Tensor.zeros(*expected_b_ho_shape) if bias else None

        self.h_prev = Tensor.zeros(*expected_h_prev_shape)

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        # LSTM cell computations
        # Check for batch size
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(self.h_prev.shape) == 2:
            self.h_prev = self.h_prev.unsqueeze(0)

        h_next_seq = []
        y_pred_seq = []

        for batch in range(x.shape[0]):
            batch_x = x[batch]
            batch_h_prev = self.h_prev[batch]
            gi = batch_x.linear(self.w_ih, self.b_ih)
            gh = batch_h_prev.linear(self.w_hh, self.b_hh)

            i_f, i_i, i_c, i_o = gi.chunk(4, 2)
            h_f, h_i, h_c, h_o = gh.chunk(4, 2)

            forgetgate = (i_f + h_f).sigmoid()
            ingate = (i_i + h_i).sigmoid()
            cellgate = (forgetgate * h_c) + (ingate * i_c.tanh())
            outgate = (i_o + h_o).sigmoid()

            batch_h_next = (outgate * cellgate.tanh()).permute(2, 1, 0)

            # Output layer computations
            batch_y_pred = batch_h_next.linear(self.w_ho, self.b_ho)

            h_next_seq.append(batch_h_next)
            y_pred_seq.append(batch_y_pred)

        h_next = Tensor.stack(h_next_seq)
        y_pred = Tensor.stack(y_pred_seq)

        self.h_prev = h_next

        return y_pred


# The generator must be more powerful than the discriminator to avoid the discriminator overpowering the generator
class Generator:
    def __init__(
        self,
        timesteps_in,
        features_in,
        timesteps_out,
        features_out,
        batch_size: Optional[int] = 1,
    ):
        self.timesteps_out = timesteps_out
        self.features_out = features_out
        self.timesteps_in = timesteps_in
        self.features_in = features_in
        self.batch_size = batch_size

        self.complexity_scalar = 8

        # Reshape layer is useful as the input shape is (batch_size, features_in, timesteps_in)
        # The model will extract more information from the time series if we expand the features_in
        self.reshape_in = Reshape((batch_size, timesteps_in, features_in))

        linear1_input_dim = features_in
        linear1_output_dim = linear1_input_dim * timesteps_in * self.complexity_scalar
        self.linear1 = Linear(linear1_input_dim, linear1_output_dim)

        gru1_input_shape = (timesteps_in * linear1_input_dim, linear1_output_dim)
        gru1_hidden_units = gru1_input_shape[0] * self.complexity_scalar
        gru1_output_shape = (batch_size, timesteps_in, gru1_hidden_units)
        debug_print(f"gru1_input_shape: {gru1_input_shape}")
        debug_print(f"gru1_hidden_units: {gru1_hidden_units}")
        debug_print(f"gru1_output_shape: {gru1_output_shape}")
        self.gru1 = GRU(
            gru1_input_shape,
            gru1_hidden_units,
            batch_size=batch_size,
            return_sequences=True,
        )

        gru2_input_shape = (
            gru1_output_shape[1],
            gru1_output_shape[2],
        )  # (num_steps, units) as return_sequences is True in GRU1
        gru2_hidden_units = gru2_input_shape[0] * self.complexity_scalar
        gru2_ouput_shape = (batch_size, timesteps_in, gru2_hidden_units)
        debug_print(f"gru2_ouput_shape: {gru2_ouput_shape}")
        self.gru2 = GRU(
            gru2_input_shape,
            gru2_hidden_units,
            batch_size=batch_size,
            return_sequences=False,
        )

        self.flatten = Flatten()

        linear2_input_dim = (
            gru2_ouput_shape[0] * gru2_ouput_shape[1] * gru2_ouput_shape[2]
        )
        linear2_output_dim = features_out * timesteps_out
        debug_print(f"linear2_input_dim: {linear2_input_dim}")
        debug_print(f"linear2_output_dim: {linear2_output_dim}")
        self.linear2 = Linear(linear2_input_dim, linear2_output_dim)

        self.reshape_out = Reshape((batch_size, timesteps_out, features_out))

        self.layers: List[Callable[[Tensor], Tensor]] = [
            self.reshape_in,
            self.linear1,
            lambda x: x.relu(),
            self.gru1,
            self.gru2,
            self.flatten,
            self.linear2,
            lambda x: x.relu(),
            self.reshape_out,
        ]

        self.trainable_variables = get_parameters(self)

    def __call__(self, x):
        # return x.sequential(self.layers)
        debug_print(f"gen input.shape: {x.shape}")
        x = self.reshape_in(x)
        debug_print(f"gen reshape_in.shape: {x.shape}")
        x = self.linear1(x)
        x = x.relu()
        debug_print(f"gen linear1.shape: {x.shape}")
        x = self.gru1(x)
        debug_print(f"gen gru1.shape: {x.shape}")
        x = self.gru2(x)
        debug_print(f"gen gru2.shape: {x.shape}")
        x = self.flatten(x)
        debug_print(f"gen flatten.shape: {x.shape}")
        x = self.linear2(x)
        x = x.relu()
        # Sigmoid activation is used to ensure that the output is between 0 and 1
        # This is because the data is normalized between 0 and 1
        x = x.sigmoid()
        debug_print(f"gen linear2.shape: {x.shape}")
        x = self.reshape_out(x)
        debug_print(f"gen reshape_out.shape: {x.shape}")
        return x


# The discriminator must be less powerful than the generator to avoid the discriminator overpowering the generator
# But the discriminator must be powerful enough to be able to distinguish between real and fake data and extract time series features
class Discriminator:
    def __init__(self, timesteps_in, features_in, batch_size: Optional[int] = 1):
        self.timesteps_in = timesteps_in
        self.features_in = features_in
        self.batch_size = batch_size

        debug_print(f"disc timesteps_in: {timesteps_in}")
        debug_print(f"disc features_in: {features_in}")
        debug_print(f"disc batch_size: {batch_size}")

        self.complexity_scalar = 32

        self.reshape_in = Reshape((batch_size, timesteps_in, features_in))

        linear1_input_dim = features_in
        linear1_output_dim = (
            linear1_input_dim * timesteps_in
        )  # Don't expand much so that the discriminator is less powerful than the generator
        self.linear1 = Linear(linear1_input_dim, linear1_output_dim)

        lstm1_input_shape = (timesteps_in * linear1_input_dim, linear1_output_dim)
        lstm1_hidden_layers = timesteps_in * features_in * self.complexity_scalar
        lstm1_hidden_output_shape = (batch_size, timesteps_in, lstm1_hidden_layers)
        lstm1_output_channels = (
            lstm1_hidden_output_shape[0]
            * lstm1_hidden_output_shape[1]
            * lstm1_hidden_output_shape[2]
        )
        self.lstm1 = LSTM(lstm1_input_shape, lstm1_hidden_layers, lstm1_output_channels)

        self.flatten = Flatten()

        linear2_input_dim = lstm1_output_channels * timesteps_in
        linear2_output_dim = 1  # Binary classification: real vs. fake
        debug_print(f"disc linear2_input_dim: {linear2_input_dim}")
        debug_print(f"disc linear2_output_dim: {linear2_output_dim}")
        self.linear2 = Linear(linear2_input_dim, linear2_output_dim)

        self.layers: List[Callable[[Tensor], Tensor]] = [
            self.reshape_in,
            self.linear1,
            lambda x: x.relu(),
            self.lstm1,
            self.flatten,
            self.linear2,
            lambda x: x.dropout(
                0.2
            ),  # Dropout to prevent overfitting and to make the discriminator less powerful than the generator
            lambda x: x.sigmoid().squeeze(),  # Use sigmoid activation for binary classification
        ]

        self.trainable_variables = get_parameters(self)

    def __call__(self, x):
        # return x.sequential(self.layers)
        debug_print(f"disc input.shape: {x.shape}")
        x = self.reshape_in(x)
        debug_print(f"disc reshape_in.shape: {x.shape}")
        x = self.linear1(x)
        x = x.relu()
        debug_print(f"disc linear1.shape: {x.shape}")
        x = self.lstm1(x)
        debug_print(f"disc lstm1.shape: {x.shape}")
        x = self.flatten(x)
        debug_print(f"disc flatten.shape: {x.shape}")
        x = self.linear2(x)
        x = x.dropout(0.2)
        x = x.sigmoid()
        debug_print(f"disc linear2.shape: {x.shape}")
        x = x.squeeze()
        debug_print(f"disc squeeze.shape: {x.shape}")
        return x


class RMSprop(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        super().__init__(params, lr)
        (
            self.alpha,
            self.eps,
            self.weight_decay,
            self.momentum,
            self.centered,
            self.foreach,
            self.maximize,
            self.differentiable,
        ) = (
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            foreach,
            maximize,
            differentiable,
        )
        self.params_with_grad = []
        self.grads = []
        for p in self.params:
            # this is needed since the grads can form a "diamond"
            if p.grad is None:
                continue
            self.params_with_grad.append(p)

            if p.grad.is_sparse:
                raise RuntimeError("RMSprop does not support sparse gradients")
            self.grads.append(p.grad)
        self.momentum_buffers = (
            [
                Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
                for t in self.params
            ]
            if self.momentum
            else []
        )
        self.grad_avgs = (
            [
                Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
                for t in self.params
            ]
            if self.centered
            else []
        )
        self.square_avgs = [
            Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
            for t in self.params
        ]
        self.step_count = 0
        assert self.foreach is None, "foreach mode is not supported"

        # Raise value error lr/epl/weight_decay/alpha are less than or equal to 0
        if lr <= 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps <= 0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if alpha <= 0:
            raise ValueError("Invalid alpha value: {}".format(alpha))

    def step(self):
        """Performs a single optimization step."""
        self.step_count += 1

        for i, param in enumerate(self.params_with_grad):
            grad = self.grads[i]
            # TODO: fix this in lazy.py, forces realization of the grad from lazy tensor
            grad.realize()
            if self.maximize:
                grad = -grad
            square_avg = self.square_avgs[i]

            if self.weight_decay > 0:
                grad += self.weight_decay * param.detach()

            square_avg.assign(
                (square_avg * self.alpha) + ((grad * grad) * (1 - self.alpha))
            )

            if self.centered:
                grad_avg = self.grad_avgs[i]
                grad_avg = grad_avg * (1 - self.alpha) + grad * self.alpha
                avg = (square_avg + grad_avg * grad_avg * -1).sqrt()
            else:
                avg = square_avg.sqrt()

            avg = avg + self.eps

            if self.momentum > 0:
                buf = self.momentum_buffers[i]
                buf = (buf * self.momentum) + (grad / avg)
                param.assign(param.detach() - (self.lr * buf))
            else:
                param.assign(param.detach() - (self.lr * (grad / avg)))

        for tensor_list in [self.momentum_buffers, self.grad_avgs, self.square_avgs]:
            self.realize(tensor_list)


# Combine the generator and discriminator into a GAN that implements wasserstein gradient penalty
""" 
TODO: Implement the following:
1.) (*) Minibatch Discrimination: This technique involves providing the discriminator with access to multiple examples in a minibatch, rather than making decisions based on individual samples. This allows the discriminator to identify if the generator is producing similar outputs for different inputs.

2.) (*) Feature Matching: Instead of optimizing the generator to fool the discriminator, optimize it to make the statistics of the generated data match the real data. This can be done by matching the intermediate layer responses in the discriminator for real and generated samples.

3.) (**) Historical Averaging: Keep track of the historical parameters of the generator and include a term in the loss function that penalizes deviation from the historical average. This encourages stability over time.

4.) (Done?) Penalize the norm of the gradients (Gradient Penalty): This is a technique used in WGAN-GP (Wasserstein GAN with Gradient Penalty) to enforce the Lipschitz constraint, which helps to stabilize the training and prevent mode collapse.

5.) (***) Use different architectures: Some GAN variants like Diverse GAN (DivGAN) and Unrolled GAN have been specifically designed to tackle mode collapse.

6.) (**) Regularization: Techniques like dropout or noise injection can also help in preventing mode collapse by adding randomness to the generator's outputs.

7.) (**) Learning Rate Scheduler: Using a learning rate scheduler that reduces the learning rate over time can help in stabilizing the training process.

8.) (*) Test Hinge Loss: Instead of using the Wasserstein loss, use the hinge loss, which is more stable and can help in preventing mode collapse.

9.) (*) Attention: Attention mechanisms can be used to help the generator focus on different parts of the input sequence at different time steps.

10.) (***) Decision Transformer: This is a transformer-based architecture that can be used to generate sequences of discrete tokens.

11.) (***) Attuned Traininer: Implement a trainer that monitors the discriminator's accuracy and adjusts the generator's/disciminator's learning rate accordingly.
"""


class WGANGP:
    def __init__(
        self,
        timesteps_in,
        features_in,
        timesteps_out,
        features_out,
        batch_size: Optional[int] = 1,
        min_batch_size: Optional[int] = 1,
    ):
        self.timesteps_in = timesteps_in
        self.features_in = features_in
        self.timesteps_out = timesteps_out
        self.features_out = features_out
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        # Only take a slice of the last ~50% of time series data for the discriminator
        self.slice_disc = int(timesteps_in // 2)
        self.generator = Generator(
            timesteps_in=timesteps_in,
            features_in=features_in,
            timesteps_out=timesteps_out,
            features_out=features_out,
            batch_size=batch_size,
        )
        self.discriminator = Discriminator(
            timesteps_in=(self.slice_disc + timesteps_out),
            features_in=features_out,
            batch_size=(batch_size * min_batch_size),
        )

        self.generator_optimizer = RMSprop(
            get_parameters(self.generator), lr=0.005, alpha=0.9
        )
        self.discriminator_optimizer = RMSprop(
            get_parameters(self.discriminator), lr=0.0075, alpha=0.75
        )

        self.loss_gen = None
        self.loss_dis = None

    def generate(self, x):
        # Concatenate the real data with the fake data
        debug_print(f"generate x.shape: {x.shape}")
        x_slice = x[:, -(self.slice_disc) :, :]
        y = x_slice.cat(self.generator(x).realize(), dim=1)
        debug_print(f"generate y.shape: {y.shape}")

        return y

    def discriminate(self, y):
        return self.discriminator(y)

    def compute_gradients_penalty(self, y_real, y_gen):
        debug_print(f"compute_gradients_penalty y_real.shape: {y_real.shape}")
        debug_print(f"compute_gradients_penalty y_gen.shape: {y_gen.shape}")
        # Compute the gradients penalty
        epsilon = Tensor.rand(y_real.shape[0], 1, 1)
        debug_print(f"compute_gradients_penalty epsilon.shape: {epsilon.shape}")
        x_hat = epsilon * y_real + (1.0 - epsilon) * y_gen
        debug_print(f"compute_gradients_penalty x_hat.shape: {x_hat.shape}")
        with set_nograd(False):
            y_hat = self.discriminate(x_hat).float()
            y_hat.backward()
        # Get the gradients of y_hat with respect to x_hat
        gradients = x_hat.grad
        gradients = gradients.reshape(self.batch_size * self.min_batch_size, -1)
        gradients_norm = (gradients * gradients).sum(axis=[1, 2]).sqrt()
        gradients_penalty = ((gradients_norm - 1.0) * (gradients_norm - 1.0)).mean()
        return gradients_penalty

    def compute_loss(self, x_real, y_real):
        # Compute the loss for the generator and discriminator
        y_gen = self.generate(x_real)

        # Logits
        y_dis_real = self.discriminate(y_real)
        y_dis_gen = self.discriminate(y_gen)

        # Gradient penalty
        d_regularizer = self.compute_gradients_penalty(y_real, y_gen)

        # Losses
        loss_gen = -y_dis_gen.mean()
        loss_dis = (y_dis_gen.mean() - y_dis_real.mean()) + d_regularizer

        self.loss_gen = loss_gen.float()
        self.loss_dis = loss_dis.float()

    # @TinyJit
    def show_accuracy(self, x_real, y_real):
        # Generate fake data
        y_fake = self.generate(x_real)
        # No idea why tensor math isn't working here
        debug_print(f"y_fake.shape: {y_fake.shape}")
        debug_print(f"y_real.shape: {y_real.shape}")
        y_fake_np = (y_fake.squeeze().numpy())[-3:]
        y_real_np = (y_real.squeeze().numpy())[-3:]
        debug_print(f"y_fake_np: {y_fake_np}")
        debug_print(f"y_real_np: {y_real_np}")
        # Root mean squared error
        rms_gen = np.sqrt(np.mean((y_real_np - y_fake_np) ** 2))
        # Compute the accuracy of the generator from the root mean squared error
        acc_gen = 100 - (rms_gen * 100)
        # Compute the accuracy of the discriminator
        disc_fake = self.discriminate(y_fake)
        disc_fake_np = disc_fake.numpy()
        debug_print(f"disc_fake_np: {disc_fake_np}")
        # Root mean squared error
        rms_dis = np.sqrt(np.mean((1 - disc_fake_np) ** 2))
        # Compute the accuracy of the discriminator from the root mean squared error
        acc_dis = 100 - (rms_dis * 100)
        avg_acc = (acc_gen + acc_dis) / 2
        print(f"acc_gen: {acc_gen} acc_dis: {acc_dis} avg_acc: {avg_acc}")

    # @TinyJit
    def train_step(self, x_real, y_real):
        with Tensor.train() and set_nograd(False):
            losses_gen = []
            losses_dis = []
            # Iterate over the data in batches
            for batch in range(len(x_real)):
                # Get the real data
                batch_x_real = x_real[batch]
                batch_y_real = y_real[batch]
                # Train the discriminator
                self.discriminator_optimizer.zero_grad()
                self.compute_loss(batch_x_real, batch_y_real)
                self.loss_dis.backward()
                self.discriminator_optimizer.step()

                # Train the generator 3 times for every 1 time the discriminator is trained
                for _ in range(3):
                    self.generator_optimizer.zero_grad()
                    self.compute_loss(batch_x_real, batch_y_real)
                    self.loss_gen.backward()
                    self.generator_optimizer.step()

                # Append the loss to the list of losses
                losses_gen.append(self.loss_gen.item())
                losses_dis.append(self.loss_dis.item())

            # Convert to numpy arrays
            losses_gen = np.array(losses_gen)
            losses_dis = np.array(losses_dis)
            # Compute the average loss for the epoch
            avg_loss_gen = np.mean(losses_gen)
            avg_loss_dis = np.mean(losses_dis)

            return avg_loss_gen, avg_loss_dis

    def __call__(self, train_x: Tensor, unsafe: Optional[bool] = False) -> Tensor:
        if not unsafe:
            # Assert that the train_x is the correct shape
            # Check shape size is 3
            assert (
                len(train_x.shape) == 3
            ), "train_x must be a 3 dimensional tensor (batch_size, timesteps_in, features_in)"
            assert (
                train_x[0] == self.batch_size
            ), "train_x must have the same batch size as the generator"
            assert (
                train_x.shape[1] == self.generator.timesteps_in
            ), "train_x must have the same number of timesteps as the generator"
            assert (
                train_x.shape[2] == self.generator.features_in
            ), "train_x must have the same number of features as the generator"
        with set_nograd(True):
            # Generate fake data
            return self.generate(train_x)

    def dummy_data(self):
        num_data_points = 16
        x_dummy = []
        y_dummy = []
        for _ in range(num_data_points):
            # Create a time sequence from 0 to 2*pi
            # Small shift for randomization
            shift = np.random.uniform(0, 2 * np.pi)
            start = 0 + shift
            end = 2 * np.pi + shift
            time = np.linspace(start, end, self.timesteps_in + self.timesteps_out)

            # Generate a sine wave based on the time sequence
            sine_wave = np.sin(time)

            # Shift and normalize the sine wave between 0 and 1
            sine_wave = (sine_wave + 1) / 2

            # Add some noise to the sine wave
            noise = np.random.normal(0, 0.1, (sine_wave.shape))

            sine_wave += noise

            # Renormalize the sine wave between 0 and 1
            sine_wave = (sine_wave - sine_wave.min()) / (
                sine_wave.max() - sine_wave.min()
            )

            sine_wave = sine_wave.reshape(
                1, (self.timesteps_in + self.timesteps_out), self.features_out
            )

            # Repeat the sine wave for each batch and feature
            this_x_dummy = Tensor(sine_wave[:, : -self.timesteps_out, :])

            # Y is going to be the next 3 time steps of X aka the next 3 values of the sine wave
            # With a slice of the time series data
            this_y_dummy = Tensor(
                sine_wave[:, -(self.timesteps_out + self.slice_disc) :, :]
            )

            x_dummy.append(this_x_dummy)
            y_dummy.append(this_y_dummy)

        # Convert to numpy arrays
        x_dummy = np.array(x_dummy)
        y_dummy = np.array(y_dummy)
        return x_dummy, y_dummy


def train_loop():
    timesteps_in = 16
    features_in = 1
    timesteps_out = 4
    features_out = features_in
    batch_size = 1
    model = WGANGP(timesteps_in, features_in, timesteps_out, features_out, batch_size)

    # Create some fake data using a sine wave
    x_dummy, y_dummy = model.dummy_data()

    debug_print("Dummy data created")
    debug_print(f"x_dummy.shape: {x_dummy[0].shape}")
    debug_print(f"y_dummy.shape: {y_dummy[0].shape}")

    num_epochs = 20
    num_steps = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        losses_gen = []
        losses_dis = []
        for step in range(num_steps):
            # Randomize the data by shuffling both x and y in the same order
            p = np.random.permutation(len(x_dummy))
            x_dummy = x_dummy[p]
            y_dummy = y_dummy[p]
            # Train the model
            loss_gen, loss_dis = model.train_step(x_dummy, y_dummy)
            losses_gen.append(loss_gen)
            losses_dis.append(loss_dis)
            print(f"step: {step} loss_gen: {loss_gen} loss_dis: {loss_dis}")
        # Convert to numpy arrays
        losses_gen = np.array(losses_gen)
        losses_dis = np.array(losses_dis)
        # Compute the average loss for the epoch
        avg_loss_gen = np.mean(losses_gen)
        avg_loss_dis = np.mean(losses_dis)
        print(f"avg_loss_gen: {avg_loss_gen} avg_loss_dis: {avg_loss_dis}")
        # Compute the accuracy of the model
        model.show_accuracy(x_dummy[0], y_dummy[0])
        # Compare the final output to the real data
        y_pred = model(x_dummy[0])
        print(f"y_pred: {(y_pred.squeeze().numpy())[-3:]}")
        print(f"y_dummy: {(y_dummy[0].squeeze().numpy())[-3:]}")
        # Discriminate on the real data vs. the fake data
        dis_real = model.discriminate(y_dummy[0])
        dis_fake = model.discriminate(y_pred)
        print(f"dis_real: {dis_real.item()}")
        print(f"dis_fake: {dis_fake.item()}")


if __name__ == "__main__":
    train_loop()
