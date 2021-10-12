from math import sqrt
import array

def quadratic(a, b, c):
  discriminant = b ** 2 - 4 * a * c
  if discriminant < 0:
    return None
  elif discriminant == 0:
    x0 = - 0, 5 * b / a
    return x0
  else:
    x1 = -0.5 * (b + sqrt(discriminant))/2 * a
    x2 = -0.5 * (b - sqrt(discriminant))/2 * a
    return min(x1, x2)

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
    self.dir = direction  # .normalize()


class Sphere:

  def __init__(self, center: Vector, radius):
    self.c = center
    self.r = radius

  def intersects(self, ray):
    dif = ray.orig - self.c
    a = ray.dir.dot_p(ray.dir)
    b = 2 * ray.dir.dot_p(dif)
    c = dif.dot_p(dif) - self.r
    point_of_intersect = quadratic(a, b, c)
    return point_of_intersect


class Camera:
  def __init__(self, width, height, focal_length, origin: Vector):
    self.width = width
    self.height = height
    self.focal_length = focal_length
    self.origin = origin

  def generate_ray(self):
    for y in range(self.height):
      for x in range(self.width):
        pixel_pos = Vector(x-(self.width/2), y - (self.height/2), self.focal_length)
        direction = pixel_pos.sub(self.origin)
        yield Ray(self.origin, direction), x, y

class Image:
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.max_val = 255
    self.pixels = [[None for _ in range(width)] for _ in range(height)]
    ppm_header = f'P3 {self.width} {self.height} {self.max_val}\n'

  def set_pixel(self, x, y, elem):
    self.pixels[y][x] = elem

  def write_ppm(self, img_file):
    def to_byte(c):
      return round(max(min(c * 255, 255), 0))
    for row in self.pixels:
      for color in row:
        img_file.write(
          "{} {} {} ".format(
            to_byte(color.x), to_byte(color.y), to_byte(color.z)
          )
        )
      img_file.write("\n")

#class Color(Vector):


class Scene:

  def __init__(self, camera, object_list: list, width, height):
    self.cam = camera
    self.ob = object_list
    self.w = width
    self.h = height

  def render(self, camera):
    infinity = 10 ** 10
    for ray, x, y in camera.generate_ray():
      min_intersection_dist = infinity
      nearest_object = None
      for object in self.object_list:
        intersection_distance = object.intersects(ray)
        if intersection_distance < min_intersection_dist:
          min_intersection_dist = intersection_distance
          nearest_object = object

    if nearest_object:
      pass
    # Draw the object



if __name__ == '__main__':
  ppm_header = f'P3 {500} {600} {255}\n'
  image = array.array('B', [0, 0, 255] * 500 * 600)
  with open('base.ppm', 'wb') as f:
    f.write(bytearray(ppm_header, 'ascii'))
    image.tofile(f)