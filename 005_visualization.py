
import tflex

def histo(name, data, step=None, buckets=None):
  if step is None:
    step = tflex.get_or_create_global_step()
  #return histogram_summary.summary_v2.histogram(name, data, step=step)
  return histogram_summary_v2.histogram(name, data, step=step, buckets=buckets)


def histv(variable, step=None):
  return histo(variable.name.split(':')[0].lstrip('model/'), variable, step=step)


import numpy as np

np.set_printoptions(suppress=True)


colormap = turbo_colormap_data = np.array([
[0.18995,0.07176,0.23217],
[0.19483,0.08339,0.26149],
[0.19956,0.09498,0.29024],
[0.20415,0.10652,0.31844],
[0.20860,0.11802,0.34607],
[0.21291,0.12947,0.37314],
[0.21708,0.14087,0.39964],
[0.22111,0.15223,0.42558],
[0.22500,0.16354,0.45096],
[0.22875,0.17481,0.47578],
[0.23236,0.18603,0.50004],
[0.23582,0.19720,0.52373],
[0.23915,0.20833,0.54686],
[0.24234,0.21941,0.56942],
[0.24539,0.23044,0.59142],
[0.24830,0.24143,0.61286],
[0.25107,0.25237,0.63374],
[0.25369,0.26327,0.65406],
[0.25618,0.27412,0.67381],
[0.25853,0.28492,0.69300],
[0.26074,0.29568,0.71162],
[0.26280,0.30639,0.72968],
[0.26473,0.31706,0.74718],
[0.26652,0.32768,0.76412],
[0.26816,0.33825,0.78050],
[0.26967,0.34878,0.79631],
[0.27103,0.35926,0.81156],
[0.27226,0.36970,0.82624],
[0.27334,0.38008,0.84037],
[0.27429,0.39043,0.85393],
[0.27509,0.40072,0.86692],
[0.27576,0.41097,0.87936],
[0.27628,0.42118,0.89123],
[0.27667,0.43134,0.90254],
[0.27691,0.44145,0.91328],
[0.27701,0.45152,0.92347],
[0.27698,0.46153,0.93309],
[0.27680,0.47151,0.94214],
[0.27648,0.48144,0.95064],
[0.27603,0.49132,0.95857],
[0.27543,0.50115,0.96594],
[0.27469,0.51094,0.97275],
[0.27381,0.52069,0.97899],
[0.27273,0.53040,0.98461],
[0.27106,0.54015,0.98930],
[0.26878,0.54995,0.99303],
[0.26592,0.55979,0.99583],
[0.26252,0.56967,0.99773],
[0.25862,0.57958,0.99876],
[0.25425,0.58950,0.99896],
[0.24946,0.59943,0.99835],
[0.24427,0.60937,0.99697],
[0.23874,0.61931,0.99485],
[0.23288,0.62923,0.99202],
[0.22676,0.63913,0.98851],
[0.22039,0.64901,0.98436],
[0.21382,0.65886,0.97959],
[0.20708,0.66866,0.97423],
[0.20021,0.67842,0.96833],
[0.19326,0.68812,0.96190],
[0.18625,0.69775,0.95498],
[0.17923,0.70732,0.94761],
[0.17223,0.71680,0.93981],
[0.16529,0.72620,0.93161],
[0.15844,0.73551,0.92305],
[0.15173,0.74472,0.91416],
[0.14519,0.75381,0.90496],
[0.13886,0.76279,0.89550],
[0.13278,0.77165,0.88580],
[0.12698,0.78037,0.87590],
[0.12151,0.78896,0.86581],
[0.11639,0.79740,0.85559],
[0.11167,0.80569,0.84525],
[0.10738,0.81381,0.83484],
[0.10357,0.82177,0.82437],
[0.10026,0.82955,0.81389],
[0.09750,0.83714,0.80342],
[0.09532,0.84455,0.79299],
[0.09377,0.85175,0.78264],
[0.09287,0.85875,0.77240],
[0.09267,0.86554,0.76230],
[0.09320,0.87211,0.75237],
[0.09451,0.87844,0.74265],
[0.09662,0.88454,0.73316],
[0.09958,0.89040,0.72393],
[0.10342,0.89600,0.71500],
[0.10815,0.90142,0.70599],
[0.11374,0.90673,0.69651],
[0.12014,0.91193,0.68660],
[0.12733,0.91701,0.67627],
[0.13526,0.92197,0.66556],
[0.14391,0.92680,0.65448],
[0.15323,0.93151,0.64308],
[0.16319,0.93609,0.63137],
[0.17377,0.94053,0.61938],
[0.18491,0.94484,0.60713],
[0.19659,0.94901,0.59466],
[0.20877,0.95304,0.58199],
[0.22142,0.95692,0.56914],
[0.23449,0.96065,0.55614],
[0.24797,0.96423,0.54303],
[0.26180,0.96765,0.52981],
[0.27597,0.97092,0.51653],
[0.29042,0.97403,0.50321],
[0.30513,0.97697,0.48987],
[0.32006,0.97974,0.47654],
[0.33517,0.98234,0.46325],
[0.35043,0.98477,0.45002],
[0.36581,0.98702,0.43688],
[0.38127,0.98909,0.42386],
[0.39678,0.99098,0.41098],
[0.41229,0.99268,0.39826],
[0.42778,0.99419,0.38575],
[0.44321,0.99551,0.37345],
[0.45854,0.99663,0.36140],
[0.47375,0.99755,0.34963],
[0.48879,0.99828,0.33816],
[0.50362,0.99879,0.32701],
[0.51822,0.99910,0.31622],
[0.53255,0.99919,0.30581],
[0.54658,0.99907,0.29581],
[0.56026,0.99873,0.28623],
[0.57357,0.99817,0.27712],
[0.58646,0.99739,0.26849],
[0.59891,0.99638,0.26038],
[0.61088,0.99514,0.25280],
[0.62233,0.99366,0.24579],
[0.63323,0.99195,0.23937],
[0.64362,0.98999,0.23356],
[0.65394,0.98775,0.22835],
[0.66428,0.98524,0.22370],
[0.67462,0.98246,0.21960],
[0.68494,0.97941,0.21602],
[0.69525,0.97610,0.21294],
[0.70553,0.97255,0.21032],
[0.71577,0.96875,0.20815],
[0.72596,0.96470,0.20640],
[0.73610,0.96043,0.20504],
[0.74617,0.95593,0.20406],
[0.75617,0.95121,0.20343],
[0.76608,0.94627,0.20311],
[0.77591,0.94113,0.20310],
[0.78563,0.93579,0.20336],
[0.79524,0.93025,0.20386],
[0.80473,0.92452,0.20459],
[0.81410,0.91861,0.20552],
[0.82333,0.91253,0.20663],
[0.83241,0.90627,0.20788],
[0.84133,0.89986,0.20926],
[0.85010,0.89328,0.21074],
[0.85868,0.88655,0.21230],
[0.86709,0.87968,0.21391],
[0.87530,0.87267,0.21555],
[0.88331,0.86553,0.21719],
[0.89112,0.85826,0.21880],
[0.89870,0.85087,0.22038],
[0.90605,0.84337,0.22188],
[0.91317,0.83576,0.22328],
[0.92004,0.82806,0.22456],
[0.92666,0.82025,0.22570],
[0.93301,0.81236,0.22667],
[0.93909,0.80439,0.22744],
[0.94489,0.79634,0.22800],
[0.95039,0.78823,0.22831],
[0.95560,0.78005,0.22836],
[0.96049,0.77181,0.22811],
[0.96507,0.76352,0.22754],
[0.96931,0.75519,0.22663],
[0.97323,0.74682,0.22536],
[0.97679,0.73842,0.22369],
[0.98000,0.73000,0.22161],
[0.98289,0.72140,0.21918],
[0.98549,0.71250,0.21650],
[0.98781,0.70330,0.21358],
[0.98986,0.69382,0.21043],
[0.99163,0.68408,0.20706],
[0.99314,0.67408,0.20348],
[0.99438,0.66386,0.19971],
[0.99535,0.65341,0.19577],
[0.99607,0.64277,0.19165],
[0.99654,0.63193,0.18738],
[0.99675,0.62093,0.18297],
[0.99672,0.60977,0.17842],
[0.99644,0.59846,0.17376],
[0.99593,0.58703,0.16899],
[0.99517,0.57549,0.16412],
[0.99419,0.56386,0.15918],
[0.99297,0.55214,0.15417],
[0.99153,0.54036,0.14910],
[0.98987,0.52854,0.14398],
[0.98799,0.51667,0.13883],
[0.98590,0.50479,0.13367],
[0.98360,0.49291,0.12849],
[0.98108,0.48104,0.12332],
[0.97837,0.46920,0.11817],
[0.97545,0.45740,0.11305],
[0.97234,0.44565,0.10797],
[0.96904,0.43399,0.10294],
[0.96555,0.42241,0.09798],
[0.96187,0.41093,0.09310],
[0.95801,0.39958,0.08831],
[0.95398,0.38836,0.08362],
[0.94977,0.37729,0.07905],
[0.94538,0.36638,0.07461],
[0.94084,0.35566,0.07031],
[0.93612,0.34513,0.06616],
[0.93125,0.33482,0.06218],
[0.92623,0.32473,0.05837],
[0.92105,0.31489,0.05475],
[0.91572,0.30530,0.05134],
[0.91024,0.29599,0.04814],
[0.90463,0.28696,0.04516],
[0.89888,0.27824,0.04243],
[0.89298,0.26981,0.03993],
[0.88691,0.26152,0.03753],
[0.88066,0.25334,0.03521],
[0.87422,0.24526,0.03297],
[0.86760,0.23730,0.03082],
[0.86079,0.22945,0.02875],
[0.85380,0.22170,0.02677],
[0.84662,0.21407,0.02487],
[0.83926,0.20654,0.02305],
[0.83172,0.19912,0.02131],
[0.82399,0.19182,0.01966],
[0.81608,0.18462,0.01809],
[0.80799,0.17753,0.01660],
[0.79971,0.17055,0.01520],
[0.79125,0.16368,0.01387],
[0.78260,0.15693,0.01264],
[0.77377,0.15028,0.01148],
[0.76476,0.14374,0.01041],
[0.75556,0.13731,0.00942],
[0.74617,0.13098,0.00851],
[0.73661,0.12477,0.00769],
[0.72686,0.11867,0.00695],
[0.71692,0.11268,0.00629],
[0.70680,0.10680,0.00571],
[0.69650,0.10102,0.00522],
[0.68602,0.09536,0.00481],
[0.67535,0.08980,0.00449],
[0.66449,0.08436,0.00424],
[0.65345,0.07902,0.00408],
[0.64223,0.07380,0.00401],
[0.63082,0.06868,0.00401],
[0.61923,0.06367,0.00410],
[0.60746,0.05878,0.00427],
[0.59550,0.05399,0.00453],
[0.58336,0.04931,0.00486],
[0.57103,0.04474,0.00529],
[0.55852,0.04028,0.00579],
[0.54583,0.03593,0.00638],
[0.53295,0.03169,0.00705],
[0.51989,0.02756,0.00780],
[0.50664,0.02354,0.00863],
[0.49321,0.01963,0.00955],
[0.47960,0.01583,0.01055]
], dtype=np.float32)


tf_turbo_data = tf.constant(turbo_colormap_data, dtype=tf.float32, name='turbo_cmap')

# The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.
# To use it with matplotlib, pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import ListedColormap").
# If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT directly.
# The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic.
# If you have a floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries.
# If you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use interpolate(). Doing the interpolation in floating point will reduce banding.
# If some of your values may lie outside the [0,1] range, use interpolate_or_clip() to highlight them.


# FROM HERE

def lerp(a, b, t):
  return (b - a) * t + a


def rerange(x, to_lo, to_hi, from_lo=None, from_hi=None):
  if from_lo is None:
    from_lo = x.min()
  if from_hi is None:
    from_hi = x.max()
  t = (x - from_lo) / (from_hi - from_lo)
  return lerp(to_lo, to_hi, t)


def interpolate(colormap, x):
  x = max(0.0, min(1.0, x))
  a = int(x*255.0)
  b = min(255, a + 1)
  f = x*255.0 - a
  return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
          colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
          colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]


def tf_interpolate(colormap, x):
  x = tf.clip_by_value(x, 0.0, 1.0)
  a = tf.cast(x*255.0, tf.int32)
  b = tf.cast(tf.minimum(255, a + 1), tf.int32)
  f = x*255.0 - tf.cast(a, x.dtype)
  A = tf.gather(colormap, a)
  B = tf.gather(colormap, b)
  f = tf.expand_dims(f, axis=-1)
  return A + (B - A) * f


def tf_interpolate64(colormap, x):
  dtype = x.dtype
  x = tf.cast(x, tf.float64)
  x = tf.clip_by_value(x, 0.0, 1.0)
  a = tf.cast(x*255.0, tf.int32)
  b = tf.cast(tf.minimum(255, a + 1), tf.int32)
  f = x*255.0 - tf.cast(a, x.dtype)
  A = tf.gather(colormap, a)
  A = tf.cast(A, tf.float64)
  B = tf.gather(colormap, b)
  B = tf.cast(B, tf.float64)
  f = tf.expand_dims(f, axis=-1)
  R = A + (B - A) * f
  R = tf.cast(R, dtype)
  return R



def interpolate_or_clip(colormap, x):
  if   x < 0.0: return [0.0, 0.0, 0.0]
  elif x > 1.0: return [1.0, 1.0, 1.0]
  else: return interpolate(colormap, x)


def is_tf_type(x):
  return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable)


def remapv(x, bot=None, top=None):
  if bot is None:
    bot = tf.reduce_min(x) if is_tf_type(x) else np.array(x).min()
  if top is None:
    top = tf.reduce_max(x) if is_tf_type(x) else np.array(x).max()
  return (x - bot) / (top - bot)


def clamp(v, min=0., max=1.):
  #return tf.minimum(tf.maximum(v, min), max)
  return tf.clip_by_value(v, min, max)


def iround(u):
  return tf.cast(tf.math.floordiv(tf.cast(u, tf.float32), 1.0), tf.int32)


def img_f32_to_u8(x):
  return tf.cast(iround(clamp(x*256.0, 0.0, 255.0)), tf.uint8)


# def cmapv(x, remap=True):
#   if remap:
#     x = remapv(x)
#   indices = iround(clamp(x*256.0, 0.0, 255.0))
#   indices = tf.expand_dims(indices, axis=-1)
#   rgb = tf.gather_nd(tf_turbo_data, indices)
#   return rgb


def cmapv(x, remap=True):
  if remap:
    x = remapv(x)
  return tf_interpolate(tf_turbo_data, x)


def cmapv2(x, remap=True):
  return tf.convert_to_tensor([1.0, 1.0, 1.0]) - cmapv(x, remap=remap)


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]



def imgv(x, remap=True, cmap=True):
  #if len(x.shape) >= 3:
  if len(x.shape) > 2:
    x = tf.reshape(x, [-1, x.shape[-1]])
  while x.shape[0] == 1:
    x = tf.squeeze(x, axis=0)
  while x.shape[-1] == 1:
    x = tf.squeeze(x, axis=-1)
  if cmap:
    rgb = cmapv(x, remap=remap)
    #rgb = cmapv2(x, remap=remap)
  else:
    rgb = tf.expand_dims(x, axis=-1)
  rgb8 = img_f32_to_u8(rgb)
  if len(rgb8.shape) <= 2:
    rgb8 = tf.tile(tf.expand_dims(rgb8, axis=0), [32, 1, 1])
  rgb8 = tf.transpose(rgb8, [1,0,2]) # [width, height, channels] to [height, width, channels]
  return tf.image.encode_png(rgb8)


def imgpath(x):
  if hasattr(x, 'name'):
    x = x.name
  return os.path.join('media', tputil.tf_sanitize_op_name(x).rsplit('_0', 1)[0] + '.png')


def imgdisk(x, path='test.png', remap=True, cmap=True, session=None):
  if session is None:
    session = tf.compat.v1.get_default_session()
  if path is None:
    path = imgpath(x)
  data = session.run(imgv(x, remap=remap, cmap=cmap))
  with open(path, 'wb') as f:
    f.write(data)


# TO HERE

from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor()

reg.fit(colormap, np.linspace(0.0, 1.0, 256))

def uncmapv(x, lo=None, hi=None):
  shape = x.shape
  x1 = reg.predict(x.reshape([-1, 3]))
  x2 = x1.reshape(x.shape[0:-1])
  if lo is None:
    lo = -1.0
  if hi is None:
    hi = 1.0
  return rerange(x2, lo, hi)

def uncmapv2(x, lo=None, hi=None):
  return uncmapv(np.array([0.0, 1.0, 0.0]) - x, lo=lo, hi=hi)



def test_cmap(x):
  lo = x.min()
  hi = x.max()
  rgb = r(cmapv(x))
  x2 = uncmapv(rgb, lo, hi)
  return np.square(x - x2).mean()

snapshot = tf1.train.load_checkpoint('../gpt-2-enc/models/117M/model.ckpt')

wpe = snapshot.get_tensor('model/wpe')

imgdisk(wpe, 'wpe_117M.png')


import PIL.Image


def img2rgb(img):
  if isinstance(img, str):
    img = PIL.Image.open(img)
  img = img.convert('RGB')
  rgb = np.array(img)
  rgb = np.transpose(rgb, [1, 0, 2])
  return rgb


wpe_rgb = img2rgb('wpe_117M.png')


wpe2 = rerange(reg.predict(wpe_rgb.reshape([-1, 3]) / 255).reshape(wpe_rgb.shape[0:2]), wpe.min(), wpe.max())

np.square(wpe - wpe2).mean()


wpe_bigger = gen_array_ops.mirror_pad(wpe, [[1024-16, 16], [256, (1600-1024)]], 'REFLECT'))


# def norm(x, std, mean):
#   return x / x.max() * std * 2 - std + mean

# embedding = torch.stack([norm((1.0*torch.arange(0, length, dtype=torch.float)) ** i, 5, 0) for i in range(1, 1+dim)], 0)


def predict(X_t, y_t, x_t, k_t):
    neg_one = tf.constant(-1.0, dtype=tf.float32)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, k_t)
    # slice the labels of these points
    y_s = tf.gather(y_t, indx)
    # take the average.
    y_s = tf.reduce_mean(y_s, 0)
    return y_s


def predict_idx(X_t, x_t, k_t):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, k_t)
    # # slice the labels of these points
    # y_s = tf.gather(y_t, indx)
    # return y_s
    return indx

def predict_idx2(X_t, x_t, k_t):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, k_t)
    # take the average.
    indx = tf.reduce_mean(tf.cast(indx, tf.float32), 0)
    # divide by dataset size.
    #indx /= tf.shape(X_t)[0]
    return indx


# Q=np.dot(normalized(B-A), normalized(P-A))*dist(A,P)*normalized(B-A)+A

def tf_normalized(A, B=None, epsilon=1e-6):
  if B is not None:
    A = B - A
  d = tf.linalg.norm(A)
  #return A / (d + epsilon)
  return tf.raw_ops.DivNoNan(x=A, y=d)


def tf_distance(A, B=None):
  if B is not None:
    A = B - A
  return tf.linalg.norm(A)


def tf_dot(A, B, axis=1):
  return tf.linalg.tensordot(A, B, axis)


def tf_closest(A, B, P):
  AB = tf_normalized(A, B)
  AP = tf_normalized(A, P)
  u = tf_dot(AB, AP)
  v = tf_distance(A, P)
  return u*v*AB + A


def predict_idx3(X_t, x_t, k_t=2):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, 2)
    # slice the labels of these points
    y_s = tf.gather(X_t, indx)
    A = y_s[0]
    B = y_s[1]
    P = x_t
    return tf_closest(A, B, P)


def predict_idx4(X_t, x_t, k_t=2):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, 2)
    # slice the labels of these points
    y_s = tf.gather(X_t, indx)
    A = y_s[0]
    B = y_s[1]
    P = x_t
    return A


def predict_idx5(X_t, x_t, k_t=2):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, 2)
    # slice the labels of these points
    y_s = tf.gather(X_t, indx)
    A = y_s[0]
    B = y_s[1]
    P = x_t
    AB = tf_normalized(A, B)
    AP = tf_normalized(A, P)
    ABt = tf_distance(A, B)
    s = tf_dot(AB, AP)
    t = tf_distance(A, P)
    # R = s*t*AB + A
    # Pu = vals[0]
    # Pv = vals[1]
    offset = s*t/ABt
    return tf.cast(indx[0], dtype=offset.dtype) - offset


def predict_idx6(X_t, x_t, k_t=2):
    x_t = tf.convert_to_tensor(x_t)
    neg_one = tf.constant(-1.0, dtype=x_t.dtype)
    X_t = tf.cast(X_t, x_t.dtype)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, 2)
    # slice the labels of these points
    y_s = tf.gather(X_t, indx)
    A = y_s[0]
    B = y_s[1]
    P = x_t
    # AB = tf_normalized(A, B)
    # AP = tf_normalized(A, P)
    # ABt = tf_distance(A, B)
    # s = tf_dot(AB, AP)
    # t = tf_distance(A, P)
    # R = s*t*AB + A
    # Pu = vals[0]
    # Pv = vals[1]
    # offset = s*t/ABt
    # return tf.cast(indx[0], dtype=offset.dtype) + offset
    return tf.cast(indx[0], dtype=P.dtype)








zz4 = tf.map_fn(lambda x: predict_idx3(X_t, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
# zz5 = tf.map_fn(lambda x: predict_idx4(X_t, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
zz6 = tf.map_fn(lambda x: predict_idx5(X_t, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
zz7 = tf.map_fn(lambda x: predict_idx6(X_t, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
# np.square(r(zz5, {X_ph: X[0:100]}) - X[0:100]).mean() / np.square(r(zz4, {X_ph: X[0:100]}) - X[0:100]).mean()


# np.square((r(zz7, {X_ph: X[0:10]}) / 255.0 * (wpe.max() - wpe.min()) + wpe.min()) - wpe[0][0:10]).mean()


X = wpe_rgb.reshape([-1, 3]) / 255


X_t = tf_turbo_data

X_ph = tf.placeholder(tf.float32, [None, 3], name="X")

#zz = tf.map_fn(lambda x: predict_idx(X_t, x, k_t), X[0:100], dtype=tf.int32, back_prop=False, parallel_iterations=50000)
zz = tf.map_fn(lambda x: predict_idx(X_t, x, k_t), X_ph, dtype=tf.int32, back_prop=False, parallel_iterations=50000)
r(zz, {X_ph: X[0:10000]})

zz2 = tf.map_fn(lambda x: predict(X_t, tf_turbo_data, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
r(zz2, {X_ph: X[0:100]})

zz3 = tf.map_fn(lambda x: predict_idx2(X_t, x, k_t), X_ph, dtype=tf.float32, back_prop=False, parallel_iterations=50000)
r(zz3, {X_ph: X[0:100]})


dummy_z = np.array([[-0.27682826, -0.58815104,  1.7036977 , -0.2206092 , -0.14080267, -1.4358327 , -0.29529673, -0.21998306,  2.3495033 , -0.2704561 , 0.67489153, -1.7079376 , -0.8530004 ,  0.47657555, -1.0244914 , -0.5066494 ,  0.40413463, -2.650805  ,  0.20753726, -0.45942673, 0.34236595,  0.78934395,  1.2019389 , -1.255674  , -0.07768833, 0.7577431 , -0.38986343,  0.03649916,  1.328297  , -0.1437277 , 0.76792073, -1.2927496 ,  2.3878598 ,  0.6853071 ,  0.11304516, -0.9645209 , -2.4931862 , -0.12763861,  0.7414209 ,  0.6020558 , -0.6050938 , -0.5639263 , -0.00988291, -1.5184546 , -0.38591796, -0.20601207,  0.18363006,  1.962519  ,  1.0850583 , -0.8571455 , -0.01302644,  0.6277824 , -0.08292245,  0.92597395, -0.11542881, 0.38168597, -2.4266438 , -1.566245  , -1.039471  , -0.940249  , 0.02636161,  0.06007156,  1.2789264 , -0.07752071,  0.44986045, 0.41236845,  1.3643209 , -1.4008445 , -0.19861189,  0.46731564, 0.06151719,  0.98628384, -0.20362067, -0.5842369 , -0.7696563 , 0.94691944,  2.3646383 , -1.1924875 ,  0.35439596,  1.2308508 , 0.17359956,  1.3657194 , -1.1731008 , -0.9649893 ,  0.87262   , -0.3879596 ,  0.12370261,  0.9923666 , -1.6314132 , -2.173692  , 1.2991096 , -0.5108776 , -0.31982934, -1.463904  ,  0.00470991, -0.18117207, -0.04366804, -1.4812558 ,  1.1272283 ,  0.5390479 , -0.03865089, -0.5393169 , -0.10081987,  0.69317263,  1.2149591 , 0.26094043,  0.71965116,  0.81613004,  1.4130529 ,  0.44307762, -0.2564097 , -0.06270383,  0.11339105,  1.2114154 ,  0.9871673 , -0.67596656, -0.34136584, -0.40325257, -1.5253726 , -0.3829709 , -1.3955748 ,  1.349158  ,  0.58127445, -0.8905083 ,  1.272159  , 0.8208986 , -0.5260699 , -1.075426  ,  0.29986796,  0.06508358, 0.3826486 ,  1.5031533 ,  1.2863646 , -0.15485081, -0.06244287, 1.1686682 ,  0.35917065,  2.2737215 ,  1.5198022 , -1.2142191 ]], dtype=np.float32)

dummy_y = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)

z = tf.convert_to_tensor(dummy_z, name='dummy_z')
y = tf.convert_to_tensor(dummy_y, name='dummy_y')




import tensorflow_hub as hub
mod = hub.Module("gs://dota-euw4a/models/biggan-256")

r([tf.global_variables_initializer(), tf.local_variables_initializer()])




graph.__dict__.update(tf.Graph().__dict__)

# Sample random noise (z) and ImageNet label (y) inputs.
deep = False
deep = True
res = 512
batch_size = 1
truncation = 0.5  # scalar truncation value in [0.02, 1.0]
z_dim = 128 if deep else 140 if res == 256 else 128
n_class = 1000
z = tf.placeholder(tf.float32, [batch_size, z_dim], name="z")
#y_index = tf.placeholder(tf.int32, [batch_size], name="y_index")
y = tf.placeholder(tf.float32, [batch_size, n_class], name="y")
#y = tf.one_hot(y_index, n_class)

interact
r = sess.run





import tensorflow_hub as hub
module = hub.Module("gs://tpu-usc1/models/biggan-deep-512" if deep else "gs://tpu-usc1/models/biggan-512")

r([tf.global_variables_initializer(), tf.local_variables_initializer()])

op_offset = len(graph.get_operations())
# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 512, 512, 3] and range [-1, 1].
#samples = module(dict(y=y, z=z, truncation=truncation))
samples = module(dict(y=y, z=z, truncation=1.0))
op_final = len(graph.get_operations())



#import tftorch as nn; reload(nn); import BigGAN; reload(BigGAN); mdl = BigGAN.BigGAN256(scope='').eval(); op_offset = len(graph.get_operations()); samples = mdl.generator(z, y); op_final = len(graph.get_operations()); samples

# import tftorch as nn; reload(nn); import BigGAN; reload(BigGAN); mdl = BigGAN.BigGAN512(scope='', disc=True).eval(); op_offset = len(graph.get_operations()); samples_plain = mdl.generator(z, y); samples = mdl.ema_generator(z, y); op_final = len(graph.get_operations()); logits = mdl.discriminator(samples, y); mdl_train = BigGAN.BigGAN512(scope='', disc=True).train(); samples_train = mdl_train.generator(z, y); samples_train_ema = mdl_train.ema_generator(z, y)

# BigGAN-256
assert deep == False
assert res == 256
from importlib import reload
import tftorch as nn; reload(nn); import BigGAN; reload(BigGAN); mdl = BigGAN.BigGAN256(scope='', disc=True).eval(); op_offset = len(graph.get_operations()); samples = mdl.ema_generator(z, y); op_final = len(graph.get_operations()); logits = mdl.discriminator(samples, y); samples

# BigGAN-512
assert deep == False
assert res == 512
from importlib import reload
import tftorch as nn; reload(nn); import BigGAN; reload(BigGAN); mdl = BigGAN.BigGAN512(scope='', disc=True).eval(); op_offset = len(graph.get_operations()); samples = mdl.ema_generator(z, y); op_final = len(graph.get_operations()); logits = mdl.discriminator(samples, y); samples

# BigGAN-deep-512
assert deep == True
import tftorch as nn; reload(nn); import BigGAN; reload(BigGAN); mdl = BigGAN.BigGANDeep512(scope='').eval(); op_offset = len(graph.get_operations()); samples = mdl.ema_generator(z, y); op_final = len(graph.get_operations()); samples

#import clipboard; import graph_to_code as gcode; reload(gcode); import biggan_256; code = [gcode.PrettyOp(op) for op in graph.get_operations()]; codes = pf(code); f = open('bg2-ops.txt', 'w'); f.write(codes); f.close(); del f

import clipboard; import graph_to_code as gcode; reload(gcode); sample_ops = [gcode.PrettyOp(op) for op in graph.get_operations()[op_offset:op_final+1]]

with open('my-sample-ops.txt', 'w') as f: [f.write(pf(op)+'\n') for op in sample_ops]





pp(tf.all_variables())

g_vars = [x for x in tf.all_variables() if 'Discriminator' not in x.name and 'module/' not in x.name]
d_vars = [x for x in tf.all_variables() if 'Discriminator' in x.name and 'module/' not in x.name]
ckpt = 'gs://tpu-usc1/models/biggan-deep-512/variables' if deep else 'gs://tpu-usc1/models/biggan-{res}/variables'.format(res=res)
saver = tf.train.Saver(var_list=g_vars); saver.restore(sess, tf.train.latest_checkpoint(ckpt));



from scipy.stats import truncnorm

def truncated_z_sample(batch_size=batch_size, z_dim=z_dim, truncation=0.5, seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
  return truncation * values



import numpy as np

def one_hot(i, num_classes):
  if isinstance(i, int):
    i = [i]
    return one_hot(i, num_classes)[0]
  a = np.array(i, dtype=np.int32)
  #num_classes = a.max()+1
  b = np.zeros((a.size, num_classes))
  b[np.arange(a.size),a] = 1
  return b





def prn(x, *args):
  print(x, *args)
  return x

def f32rgb_to_u8(x):
  return tf.cast(tf.math.floordiv(tf.clip_by_value(x*256.0, 0.0, 255.0), 1.0), tf.uint8)


def fimg2png(img):
  return tf.image.encode_png(f32rgb_to_u8((img + 1) / 2))


def datdisk(data, filename=None):
  if filename is None:
    filename = 'bgd512_0.png' if deep else 'bg{res}_0.png'.format(res=res)
  with open(filename, 'wb') as f:
    f.write(data)
  return filename


# RGB = tf.placeholder(tf.float32, [256, 256, 3], name="RGB")
# #tf.image.encode_png(f32rgb_to_u8(rerange(qq[0], 0.0, 1.0)))
# PNG = tf.image.encode_png(f32rgb_to_u8((RGB + 1) / 2))

# def fimgdisk(f32rgb, filename='bg256_0.png'):
#   idata = r(PNG, {RGB: f32rgb})
#   datdisk(idata, filename=filename)


pngs = fimg2png(samples[0])

datdisk(r(pngs, {z: truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=638), y: one_hot([598], n_class)}));

import random


datdisk(r(pngs, {z: truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=random.randint(0, 1000)), y: one_hot([random.randint(0, 1000)], n_class)}));

datdisk(r(pngs, {z: truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=prn(random.randint(0, 1000))), y: label}));

import tqdm


#one_hot(1,10)
#one_hot([1,2,3],10)


def biginterp(filename='bginterp_%03d.png', n=60, truncation=0.5, seed_a=random.randint(0, 1000), seed_b=random.randint(0, 1000), class_id=random.randint(0, 1000), class_id_b=None, start_id_a_weight=1.0, start_id_b_weight=0.0, end_id_a_weight=0.0, end_id_b_weight=1.0):
  A = truncated_z_sample(batch_size, z_dim, truncation=truncation, seed=seed_a)
  B = truncated_z_sample(batch_size, z_dim, truncation=truncation, seed=seed_b)
  if class_id_b is None:
    class_id_b = class_id
  for i, t in tqdm.tqdm(list(enumerate(np.linspace(0.0, 1.0, n)))):
    A_w = lerp(start_id_a_weight, end_id_a_weight, t)
    B_w = lerp(start_id_b_weight, end_id_b_weight, t)
    A_y = A_w*one_hot([class_id], n_class)
    B_y = B_w*one_hot([class_id_b], n_class)
    V = lerp(A, B, t)
    Y = lerp(A_y, B_y, t)
    fname = filename % i
    #datdisk(r(pngs, {z: V, y_index: [class_id]}), fname)
    datdisk(r(pngs, {z: V, y: Y}), fname)

biginterp(seed_a=547, seed_b=400, class_id=265)

biginterp(seed_a=547, seed_b=400, class_id=cockroach_id)

biginterp(seed_a=prn(random.randint(0,1000)), seed_b=prn(random.randint(0,1000)), class_id=prn(random.randint(0,1000)), class_id_b=prn(random.randint(0,1000)), n=60)


update = tf.io.write_file('gs://tpu-usc1/biggan-256.png', pngs)

while True: r(update, {z: truncated_z_sample(1, z_dim, truncation=0.5, seed=random.randint(0, 1000)), y: [one_hot(random.randint(0, 1000), n_class)]}); time.sleep(3.0);


qq = r(samples)
outi = tf.image.encode_png(f32rgb_to_u8(rerange(qq[0], 0.0, 1.0)))

with open('bg256_0.png', 'wb') as f: f.write(r(outi))



vvars = [x for x in tf.global_variables() if 'ema' not in x.name]
firstv = {x.name: nn.numel(x)[0] for x in vvars}
fv = r(firstv); pp(fv)


op_offset = len(graph.get_operations())

#truncation = 0.5
truncation = 1.0
samples = mod(dict(y=y, z=z, truncation=truncation))

op_final = len(graph.get_operations())


import clipboard; import graph_to_code as gcode; reload(gcode); all_ops = [gcode.PrettyOp(op) for op in graph.get_operations()]; sample_ops = all_ops[op_offset:op_final+1]

with open('bgdeep-sample-ops-mine.txt', 'w') as f: [f.write(pf(op)+'\n') for op in sample_ops]


with open('bgdeep-sample-ops-mine-all.txt', 'w') as f: [f.write(pf(op)+'\n') for op in all_ops]


with open('biggan-deep-512-all-ops-official.txt', 'w') as f: [f.write(pf(op)+'\n') for op in all_ops]


# >>> samples
# <tf.Tensor 'module_apply_default/G_trunc_output:0' shape=(8, 256, 256, 3) dtype=float32>
# >>> zz = r(samples)
# >>> zz
# array([[[[-0.93538266, -0.9223737 , -0.9181869 ],
#          [-0.9302768 , -0.9187247 , -0.8983205 ],
#          [-0.9735057 , -0.97091484, -0.9684918 ],
