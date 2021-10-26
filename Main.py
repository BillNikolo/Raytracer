from math import sqrt
import array

"""x-(self.width/2), y - (self.height/2)"""
# Colors
DARK_ORCHID = [191, 62, 255]
SKY_BLUE = [135, 206, 255]
ROYAL_BLUE = [72, 118, 255]
RED = [238, 44, 44]
YELLOW = [255, 255, 0]
EMERALD_GREEN = [0, 201, 87]
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

INFINITY = 10 ** 10

def quadratic(a, b, c):
  discriminant = b ** 2 - 4 * a * c
  if discriminant < 0:
    return None
  elif discriminant == 0 and a != 0:
    x0 = - 0, 5 * b / a
    return x0
  elif a != 0:
    x1 = -0.5 * (b + sqrt(discriminant))/2 * a
    x2 = -0.5 * (b - sqrt(discriminant))/2 * a
    sol = [x1, x2]
    return min([i for i in sol if i > 0])

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

  def set_z_coordinate(self, value):
    self.z = value


class Ray:

  def __init__(self, origin, direction):
    self.orig = origin
    self.dir = direction.normalize()


class Sphere:

  def __init__(self, center: Vector, radius, color):
    self.c = center
    self.r = radius
    self.col = color

  def intersects(self, ray):
    dif = ray.orig.sub(self.c)
    a = ray.dir.dot_p(ray.dir)
    b = 2 * ray.dir.dot_p(dif)
    c = dif.dot_p(dif) - self.r
    length = quadratic(a, b, c)
    return length


class Camera:
  def __init__(self, width, height, focal_distance, origin: Vector):
    self.width = width
    self.height = height
    self.focal_dis = focal_distance
    self.origin = origin
    # self.origin.set_z_coordinate(self.focal_dis)

  def generate_ray(self):
    for y in range(self.height):
      for x in range(self.width):
        pixel_pos = Vector(x-(self.width/2), y - (self.height/2), self.focal_dis)
        direction = pixel_pos.sub(self.origin)
        yield Ray(self.origin, direction), x, y


class Image:
  def __init__(self, camera):
    self.width = camera.width
    self.height = camera.height
    self.max_val = 255
    self.ppm_header = f'P6 {self.width} {self.height} {self.max_val}\n'
    self.image = array.array('B', BLACK * self.width * self.height)

  def set_pixel_color(self, x, y, color):
    index = 3 * (y * self.width + x)
    self.image[index] = color[0]
    self.image[index + 1] = color[1]
    self.image[index + 2] = color[2]

  def export_ppm(self):
    with open('base_Image.ppm', 'wb') as f:
      f.write(bytearray(self.ppm_header, "ascii"))
      self.image.tofile(f)


class Engine:

  def __init__(self, image, camera, objects):
    self.cam = camera
    self.im = image
    self.obs = objects

  def render(self):
    for ray, x, y in self.cam.generate_ray():
        self.im.set_pixel_color(x, y, self.ray_trace(ray))
    self.im.export_ppm()

  def ray_trace(self, ray):
    color = BLACK
    dist_hit, obj_hit = self.nearest_object(ray)
    if obj_hit is None:
      return color
    else:
      return obj_hit.col

  def nearest_object(self, ray):
    min_distance = INFINITY
    object_hit = None
    for ob in self.obs:
      distance = ob.intersects(ray)
      if distance is not None and distance < min_distance:
        min_distance = distance
        object_hit = ob
    return min_distance, object_hit


def main():
  objects = [Sphere(Vector(0, 0, 200), 500, ROYAL_BLUE)]
  camera = Camera(1920, 1080, 500, Vector(0, 0, 0)) # 1080p - FullHD
  image = Image(camera)
  engine = Engine(image, camera, objects)
  engine.render()

if __name__ == '__main__':
  main()
