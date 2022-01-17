
from calendar import c
from curses import KEY_A1
import math
from tkinter import N
import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from tensorflow.keras import layers
from keras import backend
from keras.models import Sequential

#hard_definition of swish
def hardswish(x):
    return x * (backend.relu(x + 3., max_value = 6.) / 6.)

#definition of swish
def swish(x):
    return x * backend.sigmoid(x)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return 


class KerasConv(Layer): 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        super().__init__()

        # input_channel in pytorch and input_shape in keras, here (c1)
        # output_channel in pytorch and filters in keras, here (c2)

        my_act = tf.identity()
        if act:
            my_act = activations.selu
        self.conv = Conv2D(
            input_shape=(c1,),  
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='same',
            use_bias=False,
            activation=my_act,
            groups=g,
        )
        self.bn = layers.BatchNormalization(c2)


    def call(self, inputs):
        con = self.conv(inputs)
        bn = self.bn(con)
        return bn


# Depth-wise convolution class
class KerasDWConv(KerasConv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        super().__init__(c1, c2, k, s, p, g=math.gcd(c1, c2), act=act, w=w)


class KerasTransformerLayer(Layer):
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = layers.Dense(units=c, use_bias=False)
        self.k = layers.Dense(units=c, use_bias=False)
        self.v = layers.Dense(units=c, use_bias=False)
        self.ma = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=c
        )
        self.fc1 = layers.Dense(units=c, use_bias=False)
        self.fc2 = layers.Dense(units=c, use_bias=False)

    def call(self, inputs):
        # return super().call(inputs, *args, **kwargs)
        inputs = self.ma(self.q(inputs), self.k(inputs), self.v(inputs))[0] + inputs
        inputs = self.fc1(inputs)
        inputs = self.fc2(inputs) + inputs
        return inputs


class KerasTransformerBlock(Layer):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = KerasConv(c1, c2)
        self.linear = layers.Dense(units=c2)
        self.tr = Sequential(*(KerasTransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def call(self, inputs):
        # return super().call(inputs, *args, **kwargs)
        b, _, w, h = inputs.shape
        p = inputs.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

    
class KerasBottleneck(Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = round(c2 * e)
        self.cv1 = KerasConv(
            c1=c1,
            c2=c_,
            k=1,
            s=1
        )
        self.cv2 = KerasConv(
            c1=c_,
            c2=c2,
            k=3,
            s=1,
            g=g
        )
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        # return super().call(inputs, *args, **kwargs)
        if self.add:
            c1 = self.cv1(inputs)
            c2 = self.cv2(c1)
            return inputs + c2
        else:
            c1 = self.cv1(inputs)
            c2 = self.cv2(c1)
            return c2

    
class KerasBottleneckCSP(Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = KerasConv(
            c1=c1,
            c2=c_,
            k=1,
            s=1
        )

        self.cv2 = Conv2D(
            input_shape=(c1,),
            filters=c_,
            k=1,
            s=1,
            use_bias=False
        )

        self.cv3 = Conv2D(
            input_shape=(c_,),
            filters=c_,
            k=1,
            s=1,
            use_bias=False
        )

        self.cv4 = KerasConv(
            c1=c_*2,
            c2=c2,
            k=1,
            s=1
        )

        self.bn = BatchNormalization(2*c_)
        self.act = activations.selu()

        self.m = Sequential(*(KerasBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    
    def call(self, inputs):
        # return super().call(inputs, *args, **kwargs)
        y1 = self.cv1(inputs)
        y1 = self.m(y1)
        y1 = self.cv3(y1)

        y2 = self.cv2(inputs)

        cun = tf.concat((y1, y2), axis=1)
        cun = self.bn(cun)
        cun = self.act(cun)

        y3 = self.cv4(cun)

        return y3



class KerasC3(Layer):
    pass
























class KerasPadding(Layer):
    def __init__(self, pad):
        super().__init__()
        self.padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class KerasBatchNormalization(Layer):
    def __init__(self, w=None):
        super().__init__()
        self.batch_norm = BatchNormalization(
            beta_initializer=initializers.Constant(w.bias.numpy()),
            gamma_initializer=initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps)

    def call(self, inputs):
        return self.batch_norm(inputs)








class KerasBottleneck(Layer):  # class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = KerasConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = KerasConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        result = None
        if self.add:
            result = inputs + self.cv2(self.cv1(inputs))
        else:
            result = self.cv2(self.cv1(inputs))

        return result