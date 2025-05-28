import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import RandomNormal

class ActorNetwork:
    def __init__(self, state_size, action_size=1, hidden_units=(300, 600), tau=0.001, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.tau = tau
        self.lr = lr

        # Build the actor and target actor models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model(tau=1.0)  # Initialize with the same weights

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(self.hidden_units[0], activation='relu')(inputs)
        x = Dense(self.hidden_units[1], activation='relu')(x)
        outputs = Dense(self.action_size, activation='tanh',
                        kernel_initializer=RandomNormal(stddev=1e-4))(x)
        return Model(inputs=inputs, outputs=outputs)

    @tf.function
    def train(self, states, action_gradients):
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
        model_weights = self.model.trainable_variables
        gradients = tape.gradient(actions, model_weights, output_gradients=-action_gradients)
        self.optimizer.apply_gradients(zip(gradients, model_weights))

    def update_target_model(self, tau=None):
        if tau is None:
            tau = self.tau
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        updated_weights = [
            tau * mw + (1 - tau) * tw for mw, tw in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(updated_weights)
