#G = globals().get('G', globals())
G = globals()

def stringp(x):
  return isinstance(x, str)

def keywordp(x):
  return stringp(x) and len(x) > 1 and x[0] == ':'

def symbolp(x):
  if keywordp(x):
    return True
  return stringp(x) and len(x) > 2 and x[0] == "|" and x[-1] == "|"

def inner(x):
  return x[1:-1]

def symbol_name(x):
  assert symbolp(x), "Expected a symbol"
  return inner(x)

def symbol_id(x):
  assert symbolp(x), "Expected a symbol"
  return inner(x).replace('-', '_')

def symbol_value(x):
  assert symbolp(x), "Expected a symbol"
  try:
    return G[x]
  except KeyError:
    if keywordp(x):
      return x
    raise

def symbol_function(x):
  assert symbolp(x), "Expected a symbol"
  return G.get(symbol_id(x))

def symbol_plist_id(x):
  assert symbolp(x), "Expected a symbol"
  return "{" + symbol_name(x) + "}"

def symbol_plist(x):
  assert symbolp(x), "Expected a symbol"
  return G.get(symbol_plist_id(x))

def plist_get(plist, property):
  if plist is None:
    return None
  n = len(plist)
  for i in range(0, n, 2):
    if plist[i] == property:
      try:
        return plist[i+1]
      except IndexError:
        return None

def plist_put(plist, property, value):
  if plist is None:
    plist = []
  n = len(plist)
  for i in range(0, n, 2):
    if plist[i] == property:
      if i+1 >= n:
        plist.append(None)
      plist[i+1] = value
      return plist
  plist.append(property)
  plist.append(value)
  return plist

def lax_plist_get(plist, property):
  n = len(plist)
  for i in range(0, n, 2):
    if equal(plist[i], property):
      try:
        return plist[i+1]
      except IndexError:
        return None

def lax_plist_put(plist, property, value):
  if plist is None:
    plist = []
  n = len(plist)
  for i in range(0, n, 2):
    if equal(plist[i], property):
      if i+1 >= n:
        plist.append(None)
      plist[i+1] = value
      return plist
  plist.append(property)
  plist.append(value)
  return plist

def setplist(symbol, plist):
  assert symbolp(symbol), "Expected a symbol"
  G[symbol_plist_id(symbol)] = plist

def get(symbol, property):
  assert symbolp(symbol), "Expected a symbol"
  pl = symbol_plist(symbol)
  return plist_get(pl, property)

def put(symbol, property, value):
  assert symbolp(symbol), "Expected a symbol"
  pl = symbol_plist(symbol)
  pl = plist_put(pl, property, value)
  setplist(symbol, pl)
  return value

def y_len(l):
  n = -1
  for k, v in y_for(l):
    if isinstance(k, int):
      if n < k:
        n = k
  n += 1
  return n

def y_get(l, key, unset=None):
  if isinstance(key, int) and key < 0:
    n = y_len(l)
    key = clamp(key + n, 0, n - 1)
  for k, v in y_for(l):
    if k == key:
      return v
  return unset

def y_put(l, key, val):
  if isinstance(key, int) and key < 0:
    n = y_len(l)
    key = clamp(key + n, 0, n - 1)
  r = []
  seen = False
  n = -1
  for k, v in y_for(l):
    if k == key:
      v = val
      seen = True
    if isinstance(k, str):
      r.append(keyword(k))
    elif isinstance(k, int):
      if n < k:
        n = k
    r.append(v)
  n += 1
  if not seen:
    k = key
    v = val
    if isinstance(k, str):
      r.append(keyword(k))
    else:
      while n < k:
        r.append(None)
        n += 1
    r.append(v)
  return r
  

def make_symbol(x):
  assert stringp(x), "Expected a string"
  return "|" + x + "|"

def car(x):
  return x[0]

def cdr(x):
  try:
    return x[1:]
  except TypeError:
    return x

def cddr(x):
  try:
    return x[2:]
  except TypeError:
    return x

def y_key(x):
  if keywordp(x):
    return x[1:]
  else:
    return x

def y_next(h):
  if keywordp(car(h)):
    return cddr(h)
  return cdr(h)

from collections import abc

def either(x, *ys):
  if null(x) and len(ys) > 0:
    return either(*ys)
  return x

def Or(*args):
  if len(args) <= 0:
    return []
  if len(args) <= 1:
    return args[0]
  x = args[0]
  if not nil(x):
    return x
  else:
    return Or(*args[1:])

def awaitable(x):
  return isinstance(x, abc.Awaitable)

async def AND(x, *args):
  if len(args) <= 0:
    return x

def orf (*fns):
  def fn(*args):
    def self(fs):
      if t(fs):
        return eitherf(apply(car(fs), args, kws), lambda: self(cdr(fs)))



def eitherf(x, body):
  if null(x):
    return body()
  if not null(x) and len(ys) > 0:
    return either(*ys)
  return x

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

def tuples(l, n):
  return list(chunks(l, n))

def pair(l):
  return tuples(l, 2)

def dbind(n, l):
  for v in l:
    v = list(v)
    while len(v) < n:
      v.append(None)
    while len(v) > n:
      v.pop()
    yield v

def replace(x, *subs):
  for a, b in dbind(2, pair(subs)):
    if equal(x, a):
      x = b
  return x

def inner(x):
  return x[1:-1]

def set(k, v):
  assert symbolp(k), "Expected a symbol"
  G[k] = v
  return v

def keyword(x):
  if keywordp(x):
    return x
  if stringp(x) and len(x) > 0:
    return ":" + x

def setq(k, v):
  return set(make_symbol(k), v)

def rep(x):
  x = repr(x)
  if x in ['()', '[]', 'None']:
    return 'nil'
  return x

def equal(a, b):
  if inspect.isfunction(a) and inspect.isfunction(b) and a.__qualname__ == b.__qualname__:
    return True
  return rep(a) == rep(b)

def cons(a, b=None):
  assert iterable(b) or null(b)
  return [a, *([] if not iterable(b) else b)]
  
def push(element, listname):
  symbol = make_symbol(listname)
  G[symbol] = cons(element, G.get(symbol, []))
  return G[symbol]
  

def add_to_list(symbol, element, *, append=None, compare_fn=None):
  assert symbolp(symbol), "Expected a symbol"
  if compare_fn is None:
    compare_fn = equal
  l = G.get(symbol, [])
  for x in l:
    if compare_fn(x, element):
      return l
  if yes(append):
    l.append(element)
  else:
    l = cons(element, l)
  set(symbol, l)
  return l

def named(name, value, *, qualname=None):
  value.__name__ = name
  value.__qualname__ = name if qualname is None else qualname
  return value

def defalias(name, definition, *, doc=None):
  assert symbolp(name), "Expected a symbol"
  if doc is not None:
    definition.__doc__ = doc
  G[symbol_id(name)] = named(name, definition)
  return name

import inspect

def eval(x):
  if symbolp(x):
    return symbol_value(x)
  return x

def call(f, *args, **kws):
  if symbolp(f):
    f = symbol_function(f)
  return f(*args, **kws)

def run_hooks(*hookvars):
  for hookvar in hookvars:
    run_hook_with_args(hookvar)

def run_hook_with_args(hook, *args, **kws):
  if symbolp(hook):
    hook = symbol_value(hook)
  if inspect.isfunction(hook):
    hook = [hook]
  if hook is not None:
    for fn in hook:
      call(fn, *args, **kws)

def run_hook_with_args_until_success(hook, *args, **kws):
  if symbolp(hook):
    hook = symbol_value(hook)
  if inspect.isfunction(hook):
    hook = [hook]
  if hook is not None:
    for fn in hook:
      result = call(fn, *args, **kws)
      if result is not None:
        return result

def run_hook_with_args_until_failure(hook, *args, **kws):
  if symbolp(hook):
    hook = symbol_value(hook)
  if inspect.isfunction(hook):
    hook = [hook]
  if hook is not None:
    for fn in hook:
      result = call(fn, *args, **kws)
      if no(result):
        return result
  return True
    

def y_for(h, upto=None):
  if inspect.ismodule(h):
    h = vars(h)
  if isinstance(h, abc.Mapping):
    for k, v in h.items():
      yield k, v
    return
  try:
    it = iter(h)
  except TypeError:
    return
  try:
    i = -1
    while True:
      v = next(it)
      if keywordp(v):
        k = y_key(v)
        v = next(it)
        yield k, v
      else:
        if upto is not None:
          if i >= upto:
            return
        i += 1
        yield i, v
  except StopIteration:
    pass

def maybe_int(x):
  if string63(x):
    try:
      return int(x)
    except ValueError:
      pass
  return x

def isa(x, *types):
  return isinstance(x, types)

def clamp(n, lo=None, hi=None):
  if lo is not None and n < lo:
    return lo
  if hi is not None and n > hi:
    return hi
  return n

def iterable(x):
  return isinstance(x, abc.Iterable)

# def no(x):
#   if x is None:
#     return True
#   if x is False:
#     return True
#   if iterable(x) and len(x) == 0:
#     return True
#   return False

# def yes(x):
#   return not no(x)

def null(x):
  return x is None

def nil(x):
  return null(x) or none63(x)

def t(x):
  return not nil(x)

def no(x):
  return null(x) or x is False

def yes(x):
  return not no(x)

def at(x, i):
  if nil(x):
    return x
  return x[i]

def cut(x, lo=None, hi=None):
  if nil(x):
    return x
  return x[lo:hi]

def hd(x):
  return at(x, 0)

def tl(x):
  return cut(x, 1)


def only(f):
  def fn(*args, **kws):
    if t(hd(args)):
      return f(*args, **kws)

def iterate(x, upto=None):
  if hasattr(x, 'items'):
    for k, v in x.items():
      i = number(k)
      if nil(i):
        yield [k, v]

def cut(x, i):
  if i < 0:
    return x
  return 
  if nil(x):
    return x

def length(x, upto=None):
  if nil(x):
    return 0
  # if upto is None:
  #   return len(x)
  # else:
  if True:
    it = iter(x)
    i = 0
    try:
      while True:
        if is63(upto):
          if i > upto:
            return i
        next(it)
        i += 1
    except StopIteration:
      return i


def many63(x): return length(x, 1) == 2

def some63(x): return length(x, 0) == 1

def none63(x): return length(x, 0) == 0

def one63(x): return length(x, 1) == 1

def two63(x): return length(x, 2) == 2

def either(x, *ys):
  if x is None:
    if len(ys) > 0:
      return either(*ys)
  return x

def number(x):
  try:
    return int(x)
  except ValueError:
    try:
      return float(x)
    except ValueError:
      pass

def maybe_number(x):
  r = number(x)
  return x if r is None else r
