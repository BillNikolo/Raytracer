from math import sqrt
import array, random

INFINITY = 10 ** 10

def quadratic(a, b, c):
  discriminant = b ** 2 - 4 * a * c
  if discriminant < 0:
    return None
  elif discriminant == 0 and a != 0:
    x0 = - 0.5 * b / a
    return x0
  elif a != 0:
    x1 = -0.5 * (b + sqrt(discriminant))/2 * a
    x2 = -0.5 * (b - sqrt(discriminant))/2 * a
    sol = [x1, x2]
    if x1 > 0 or x2 > 0:
      return min([i for i in sol if i > 0])
    else:
      return None

class Vector:

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def rgb_mul(self, V):
    return Vector(self.x * V.x, self.y * V.y, self.z * V.z)

  def magnitude(self):
    return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

  def add(self, V):
    return Vector(self.x + V.x, self.y + V.y, self.z + V.z)

  def sub(self, V):
    return Vector(self.x - V.x, self.y - V.y, self.z - V.z)

  def mult_scalar(self, s):
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

  def reverse(self):
    return Vector(-self.x, -self.y, -self.z)


class Ray:

  def __init__(self, origin, direction):
    self.orig = origin
    self.dir = direction.normalize()


class Material:

  def __init__(self, color):
    self.color = color


class Diffuse(Material):

  def __init__(self, color):
    super().__init__()

  def diff_reflections(self, ray, distance, object, new_rays: int):
    rays_list = []
    new_color = self.color
    for i in range(new_rays):
      temp_ray = self.reflected_ray(ray, distance, object)
      rays_list.append(temp_ray)
    return rays_list

  def reflected_ray(self, ray, distance, object):
    intersection_point = (ray.dir.mult_scalar(distance)).add(ray.orig)
    normal = object.normal(intersection_point).normalize()
    rnd_vec = Vector(intersection_point.x + (random.random()-0.5),
                     intersection_point.y + (random.random()-0.5),
                     intersection_point.z + (random.random()-0.5)).normalize()
    if rnd_vec.dot_p(normal) < 0:
      rnd_vec = rnd_vec.reverse()
    return Ray(intersection_point, rnd_vec)


class Sphere:

  def __init__(self, center: Vector, radius, material: Diffuse, isitalight: bool):
    self.c = center
    self.r = radius
    self.material = material
    self.isitalight = isitalight

  def intersects(self, ray):
    dif = ray.orig.sub(self.c)
    a = ray.dir.dot_p(ray.dir)
    b = 2 * ray.dir.dot_p(dif)
    c = dif.dot_p(dif) - self.r * self.r
    length = quadratic(a, b, c)
    return length

  def normal(self, V):
    return V.sub(self.c)


class Camera:
  def __init__(self, width, height, focal_distance):
    self.width = width
    self.height = height
    self.focal_dis = focal_distance
    self.origin = Vector(0, 0, -self.focal_dis)

  def generate_ray(self):
    for y in range(self.height):
      for x in range(self.width):
        pixel_pos = Vector(x-(self.width/2), y - (self.height/2), 0)
        direction = pixel_pos.sub(self.origin)
        yield Ray(self.origin, direction), x, -y


class Image:
  def __init__(self, camera):
    self.width = camera.width
    self.height = camera.height
    self.max_val = 255
    self.ppm_header = f'P6 {self.width} {self.height} {self.max_val}\n'
    self.image = array.array('B', WHITE_BG * self.width * self.height)

  def set_pixel_color(self, x, y, color):
    index = 3 * (y * self.width + x)
    self.image[index] = int(color.x * 255)
    self.image[index + 1] = int(color.y * 255)
    self.image[index + 2] = int(color.z * 255)

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

  def ray_trace(self, ray, depth=0):
    bg_color = BLACK
    if depth > 4:
      return bg_color
    intersection_dist, obj_hit, new_color = self.nearest_object(ray)
    if obj_hit is None:
      return bg_color
    elif not obj_hit.isitalight:
      return new_color
    else:
      new_rays, new_color = obj_hit.material.diff_reflections(ray, intersection_dist, obj_hit, 4)
      for ray in new_rays:
        self.ray_trace(ray, depth+1)
      return new_color


  def nearest_object(self, ray):
    min_distance = INFINITY
    object_hit = None
    spot_color = BLACK
    for ob in self.obs:
      distance = ob.intersects(ray)
      if distance is not None and distance < min_distance:
        min_distance = distance
        object_hit = ob
    return min_distance, object_hit, spot_color


# Colors
DARK_ORCHID = Vector(0.5, 0.2, 1)
SKY_BLUE = Vector(0, 0.3, 0.9)
ROYAL_BLUE = Vector(0.5, 0.3, 0.9)
RED = Vector(1, 0.3, 0)
YELLOW = Vector(0.2, 1, 0)
EMERALD_GREEN = Vector(0, 0.9, 0)
BLACK = Vector(0, 0, 0)
WHITE = Vector(1, 1, 1)
WHITE_BG = [255, 255, 255]


def main():
  objects = [Sphere(Vector(150, 0, 400), 150, ROYAL_BLUE, True), Sphere(Vector(600, 500, 1000), 220, YELLOW, False),
             Sphere(Vector(-200, -1000, 1000), 800, EMERALD_GREEN, True)]
  camera = Camera(1280, 720, 500)  # 720p
  image = Image(camera)
  engine = Engine(image, camera, objects)
  engine.render()


if __name__ == '__main__':
  main()
