
from ast import Mod
from calendar import c
from curses import KEY_A1
import math
from tkinter import N
import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from tensorflow.keras import layers
from keras import backend
from keras.models import Sequential, Model
from tensorflow import keras
import warnings
from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from tensorflow.keras import mixed_precision as amp
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

import yaml
from PIL import Image


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


class KerasConv(Model): 
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


class KerasTransformerLayer(Model):
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


class KerasTransformerBlock(Model):
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

    
class KerasBottleneck(Model):
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

    
class KerasBottleneckCSP(Model):
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



class KerasC3(Model):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = KerasConv(c1, c_, 1, 1)
        self.cv2 = KerasConv(c1, c_, 1, 1)
        self.cv3 = KerasConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = Sequential(*(KerasBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(tf.concat((self.m(self.cv1(x)), self.cv2(x)), dim=1))




class KerasC3TR(KerasC3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = KerasTransformerBlock(c_, c_, 4, n)


class KerasC3SPP(KerasC3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = KerasSPP(c_, c_, k)


class KerasC3Ghost(KerasC3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = Sequential(*(KerasGhostBottleneck(c_, c_) for _ in range(n)))




class KerasSPP(Model):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = KerasConv(c1, c_, 1, 1)
        self.cv2 = KerasConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = [tf.keras.layers.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]

    def call(self, x):
        x = self.cv1(x)
        
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 1))




class KerasSPPF(Model):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = KerasConv(c1, c_, 1, 1)
        self.cv2 = KerasConv(c_ * 4, c2, 1, 1)
        self.m = tf.keras.layers.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def call(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 1))

            
class KerasFocus(Model):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = KerasConv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def call(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(tf.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))



class KerasGhostConv(Model):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = KerasConv(c1, c_, k, s, None, g, act)
        self.cv2 = KerasConv(c_, c_, 5, 1, None, c_, act)

    def call(self, x):
        y = self.cv1(x)
        return tf.concat([y, self.cv2(y)], 1)

class KerasGhostBottleneck(Model):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = Sequential(KerasGhostConv(c1, c_, 1, 1),  # pw
                                  KerasDWConv(c_, c_, k, s, act=False) if s == 2 else tf.Identity(),  # dw
                                  KerasGhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = Sequential(KerasDWConv(c1, c1, k, s, act=False),
                                      KerasConv(c1, c2, 1, 1, act=False)) if s == 2 else tf.Identity()

    def call(self, x):
        return self.conv(x) + self.shortcut(x)



class KerasContract(Model):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def call(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        shape = (b, c * s * s, h // s, w // s)
        return tf.reshape(x, shape)  # x(1,256,40,40)


class KerasExpand(Model):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def call(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        shape = (b, c // s ** 2, h * s, w * s)
        return tf.reshape(x, shape)



class KerasConcat(Model):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def call(self, x):
        return tf.concat(x, self.d)






class KerasDetectMultiBackend(Model):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pb', device=None, dnn=False, data=None):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   OpenVINO:               *.xml
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   TensorFlow Edge TPU:    *_edgetpu.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        from experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.engine', '.tflite', '.pb', '', '.mlmodel', '.xml']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml, xml = (suffix == x for x in suffixes)  # backends
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names


        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = tf.convert_to_tensor(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():  # Edge TPU
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli  # install https://coral.ai/software/#edgetpu-runtime
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def call(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        

        if self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = y[list(y)[-1]]  # last output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.tflite:  # Lite
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h

        y = tf.Tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.jit or self.onnx or self.engine:  # warmup types
            if isinstance(self.device, tf.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = tf.zeros(*imgsz).to(self.device).type(tf.half if half else tf.float32)  # input image
                self.forward(im)  # warmup







class KerasAutoShape(Model):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, KerasDetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    
    def call(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else tf.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, tf.Tensor):
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = tf.convert_to_tensor(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return KerasDetections(imgs, y, files, t, self.names, x.shape)



class KerasDetections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [tf.Tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [KerasDetections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class KerasClassify(Model):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = GlobalAveragePooling2D(1)  # to x(b,c1,1,1)
        self.conv = Conv2D(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def call(self, x):
        z = tf.concat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

        
