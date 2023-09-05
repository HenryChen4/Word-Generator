import numpy as np
from activations import tanh, softmax, del_S_del_z

# Predict next letter of "hello"
# given a letter bank ['h', 'e', 'l', 'o']
# Each letter is represented as 1s and 0s
# ie. 'h' = [1, 0, 0, 0], 'l' = [0, 0, 1, 0]

class Recurrent_Layer:
    def __init__(self, time_steps, hidden_state_units):
        self.hidden_states = []
        self.hidden_states.append(np.zeros((hidden_state_units, 1)))
        for i in range(time_steps):
            self.hidden_states.append(np.zeros((hidden_state_units, 1)))
        self.hidden_states = np.array(self.hidden_states)

        self.W_hh = None
        self.W_hx = None
        self.W_yh = None
        self.b_h = None
        self.b_y = None

    def summarize(self):
        print(self.hidden_states.shape)

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.X_train = None
        self.Y_train = None

    def initialize(self, X_train, Y_train, seed):
        np.random.seed(seed)

        self.X_train = X_train
        self.Y_train = Y_train

        for layer in self.layers:
            W_hh_shape = (layer.hidden_states.shape[1], layer.hidden_states.shape[1])
            W_hx_shape = (layer.hidden_states.shape[1], X_train.shape[1])
            W_yh_shape = (Y_train.shape[1], layer.hidden_states.shape[1])
            
            b_h_shape = (layer.hidden_states.shape[1], 1)
            b_y_shape = (Y_train.shape[1], 1)

            layer.W_hh = np.random.randn(W_hh_shape[0], W_hh_shape[1])
            layer.W_hx = np.random.randn(W_hx_shape[0], W_hx_shape[1])
            layer.W_yh = np.random.randn(W_yh_shape[0], W_yh_shape[1])

            layer.b_h = np.random.randn(b_h_shape[0], 1)
            layer.b_y = np.random.randn(b_y_shape[0], 1)
    
    def forward_propagation(self, a_in, h_activ, y_activ):
        all_preds = []
        for layer in self.layers:
            all_preds = []

            total_timesteps = layer.hidden_states.shape[0]

            # updating the hidden states
            for t in range(1, total_timesteps):
                z_h = np.matmul(layer.W_hh, layer.hidden_states[t-1]) + np.matmul(layer.W_hx, a_in[t-1]) + layer.b_h
                h_t = h_activ(z_h)
                layer.hidden_states[t] = h_t

            # making a prediction for all time steps
            for t in range(1, total_timesteps):
                z_y = np.matmul(layer.W_yh, layer.hidden_states[t]) + layer.b_y
                preds = y_activ(z_y)
                all_preds.append(preds)

            a_in = all_preds

        return np.array(all_preds)
    
    def back_propagation(self):
        print("working on it")

    # prints out entire model structure
    def summarize(self):
        print("X train")
        print("-------")
        for x in self.X_train:
            print(x)
            print("-----")

        print("*******************")

        for layer in self.layers:
            print("W_hx")
            print("----")
            for w in layer.W_hx:
                print(w)

        print("*******************")

        for layer in self.layers:
            print("W_hh")
            print("----")
            for w in layer.W_hh:
                print(w)

        print("*******************")

        for layer in self.layers:
            print("b_h")
            print("----")
            for b in layer.b_h:
                print(b)

        print("*******************")

        for layer in self.layers:
            print("Hidden states")
            print("-------------")
            for t in layer.hidden_states:
                print(t)
                print("-----")
        
        print("*******************")

        for layer in self.layers:
            print("W_yh")
            print("----")
            for w in layer.W_yh:
                print(w)

        print("*******************")

        for layer in self.layers:
            print("b_y")
            print("----")
            for b in layer.b_y:
                print(b)

        print("*******************")

        predictions = self.forward_propagation(self.X_train, tanh, softmax)
        print("Predictions")
        print("-----------")
        for pred in predictions:
            print(pred)
            print('----')

        print("*******************")
    
X_train = np.array([[[1], [0], [0], [0]], [[0], [1], [0], [0]], [[0], [0], [1], [0]], [[0], [0], [1], [0]]])
Y_train = np.array([[[0], [1], [0], [0]], [[0], [0], [1], [0]], [[0], [0], [1], [0]], [[0], [0], [0], [1]]])

seed = 100

layer_1 = Recurrent_Layer(time_steps=4, hidden_state_units=3)

model = Model([layer_1])
model.initialize(X_train, Y_train, seed)

model.summarize()

# predictions = model.forward_propagation(X_train, tanh, softmax)
# print(predictions)