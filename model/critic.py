import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras.optimizers import Adam

class CriticNetwork:
    def __init__(self, state_size, action_size=1, hidden_units=(300, 600), tau=0.001, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.tau = tau
        self.lr = lr

        # Build main and target critic models
        self.model, self.state_input, self.action_input = self.build_model()
        self.target_model, _, _ = self.build_model()
        self.update_target_model(tau=1.0)  # Initial sync

        # Optimizer
        self.optimizer = Adam(learning_rate=self.lr)

    def build_model(self):
        state_input = Input(shape=(self.state_size,))
        state_h1 = Dense(self.hidden_units[0], activation="relu")(state_input)
        state_h2 = Dense(self.hidden_units[1], activation="linear")(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(self.hidden_units[1], activation="linear")(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(self.hidden_units[1], activation="relu")(merged)
        output_layer = Dense(1, activation="linear")(merged_h1)

        model = Model(inputs=[state_input, action_input], outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")
        return model, state_input, action_input

    @tf.function
    def get_action_gradients(self, states, actions):
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions], training=True)
        return tape.gradient(q_values, actions)

    def update_target_model(self, tau=None):
        if tau is None:
            tau = self.tau
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        updated_weights = [tau * mw + (1 - tau) * tw for mw, tw in zip(main_weights, target_weights)]
        self.target_model.set_weights(updated_weights)
