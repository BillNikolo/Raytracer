from math import sqrt

def quadratic(a,b,c, x0, x1):
  discriminant = b ** 2 - 4 * a * c
  if discriminant < 0:
    return False
  elif discriminant == 0:
    x1 = - 0,5 * b / a
    x2 = x1
    return True
  else:
    x1 = -0.5 * (b + sqrt(discriminant))/ 2 * a
    x2 = -0.5 * (b - sqrt(discriminant))/ 2 * a
    if x1 > x2:
      x1, x2 = x2, x1
    return True

class Vector:

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def magnitude(self):
    return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

  def add(self, V):
    return Vector(self.x + V.x, self.y + V.y, self.z + V.z)

  def sub(self, V):
    return Vector(self.x - V.x, self.y - V.y, self.z - V.z)

  def multi_scalar(self, s):
    return Vector(self.x * s, self.y * s, self.z * s)

  def dot_p(self, V):
    return self.x * V.x + self.y * V.y + self.z * V.z

  def cross_p(self, V):
    return Vector(self.y * V.z - self.z * V.y,
                  self.z * V.x - self.x * V.z,
                  self.x * V.y - self.y * V.x)

  def length(self):
    return sqrt(pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2))

  def normalize(self):
    length = self.length()
    return Vector(self.x / length, self.y / length, self.z / length)

  def represent(self):
    out = str(self.x) + "i "
    if self.y >= 0:
      out += "+ "
    out += str(self.y) + "j "
    if self.z >= 0:
      out += "+ "
    out += str(self.z) + "z"
    return out


class Ray:

  def __init__(self, origin, direction):
    self.orig = origin
    self.dir = direction.normalize()


class Sphere:

  def __init__(self, center, radius):
    self.c = center
    self.r = radius

  def intersects(self, ray):
    t0 = 0
    t1 = 0
    dif = ray.orig - self.c
    a = ray.dir.dot_p(ray.dir)
    b = 2 * ray.dir.dot_p(dif)
    c = dif.dot_p(dif) - self.r
    if not(quadratic(a, b, c, t0, t1)):
      return True


class Scene:

  def __init__(self, camera, object, width, height):
    self.cam = camera
    self.ob = object
    self.w = width
    self.h = height