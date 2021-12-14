from math import sqrt
import random, pygame, time

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

  def rgb_mult(self, V):
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

  def extract_color(self):
    return (min(int(self.x * 255), 255),
            min(int(self.y * 255), 255),
            min(int(self.z * 255), 255))

  def average_of_two(self, V):
    if V == Vector(0, 0, 0) or V == Vector(1, 1, 1):
      return V
    else:
      return Vector((self.x + V.x)/2, (self.y + V.y)/2, (self.z + V.z)/2)


class Ray:

  def __init__(self, origin, direction):
    self.orig = origin
    self.dir = direction.normalize()


class Material:

  def __init__(self, color, isitalight: bool):
    self.color = color
    self.isItALight = isitalight

  def get_sample_rays(self, point, normal):
    raise NotImplemented


class Diffuse(Material):

  def __init__(self, color, isitalight):
    super().__init__(color, isitalight)

  def get_sample_rays(self, point, normal):
    for _ in range(4):
      rnd_vec = Vector(random.random()-0.5,
                       random.random()-0.5,
                       random.random()-0.5).normalize()
      cos_angle = rnd_vec.dot_p(normal)
      if cos_angle < 0:
        rnd_vec = rnd_vec.reverse()
        cos_angle *= -1
      weights = Vector(cos_angle * self.color.x,
                       cos_angle * self.color.y,
                       cos_angle * self.color.z,)
      yield Ray(point, rnd_vec), weights


class Sphere:

  def __init__(self, center: Vector, radius, material: Diffuse):
    self.c = center
    self.r = radius
    self.material = material

  def intersects(self, ray):
    L = self.c.sub(ray.orig)
    tca = L.dot_p(ray.dir)
    if tca < 0:
      return None

    d2 = L.dot_p(L) - tca * tca

    if d2 > self.r * self.r:
      return None
    thc = sqrt(self.r * self.r - d2)
    t0 = tca - thc
    t1 = tca + thc
    return min(t0, t1)

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
        x_cord = (x - (self.width/2)) + round(random.random(), 2)
        y_cord = ((self.height/2) - y) + round(random.random(), 2)
        pixel_pos = Vector(x_cord, y_cord, 0)
        direction = pixel_pos.sub(self.origin)
        yield Ray(self.origin, direction), x, y


class Image:
  def __init__(self, camera):
    pygame.init()
    self.width = camera.width
    self.height = camera.height
    self.window = pygame.display.set_mode((self.width, self.height))
    pygame.display.set_caption("Raytracer")
    self.window.fill(BLACK.extract_color())
    self.update_counter = 0
    self.sample_matrix = []
    self.max_color_value = 1.0
    for _ in range(self.width):
      l = []
      self.sample_matrix.append(l)
      for i in range(self.height):
        l.append(Vector(0, 0, 0))

    """self.width = camera.width
    self.height = camera.height
    self.max_val = 255
    self.ppm_header = f'P6 {self.width} {self.height} {self.max_val}\n'
    self.image = array.array('B', WHITE_BG * self.width * self.height)"""

  def set_pixel_color(self, x, y, color):
    self.sample_matrix[x][y] = self.sample_matrix[x][y].add(color)
    temp = max(self.sample_matrix[x][y].x, self.sample_matrix[x][y].y, self.sample_matrix[x][y].z)
    if temp > self.max_color_value:
      self.max_color_value = temp


    """index = 3 * (y * self.width + x)
    self.image[index] = min(int(color.x * 255), 255)
    self.image[index + 1] = min(int(color.y * 255), 255)
    self.image[index + 2] = min(int(color.z * 255), 255)"""

  def export_final_image(self):
    for y in range(self.height):
      for x in range(self.width):
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            pygame.quit()
            print("Sceen closed")
        gamma_correction = Vector(sqrt(self.sample_matrix[x][y].x/self.max_color_value),
                                  sqrt(self.sample_matrix[x][y].y/self.max_color_value),
                                  sqrt(self.sample_matrix[x][y].z/self.max_color_value))
        self.window.set_at((x, y), gamma_correction.extract_color())
        self.update_counter += 1
        if self.update_counter % 200 == 0:
          pygame.display.update()
          time.sleep(0.01)
    pygame.display.update()

    """with open('base_Image.ppm', 'wb') as f:
      f.write(bytearray(self.ppm_header, "ascii"))
      self.image.tofile(f)"""


class Engine:

  def __init__(self, image, camera, objects):
    self.cam = camera
    self.im = image
    self.obs = objects

  def render(self):
    for ray, x, y in self.cam.generate_ray():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
              pygame.quit()
              print("Sceen closed")
        self.im.set_pixel_color(x, y, self.ray_trace(ray))
    self.im.export_final_image()

  def ray_trace(self, ray, depth=0):
    if depth > 2:    # Maximum Level of Depth
      return BLACK
    intersection_dist, obj_hit = self.nearest_object(ray)
    if obj_hit is None:
      return BLACK
    elif obj_hit.material.isItALight:
      return obj_hit.material.color
    else:
      intersection_point = (ray.dir.mult_scalar(intersection_dist)).add(ray.orig)
      normal = obj_hit.normal(intersection_point).normalize()
      color = Vector(0, 0, 0)
      num_samples = 0
      for sample_ray, weights in obj_hit.material.get_sample_rays(intersection_point, normal):
        sample_color = self.ray_trace(sample_ray, depth + 1)
        sample_color = sample_color.rgb_mult(weights)
        color = color.add(sample_color)
        num_samples += 1
      color = color.mult_scalar(1 / num_samples)
      return color

  def nearest_object(self, ray):
    min_distance = INFINITY
    object_hit = None
    for ob in self.obs:
      distance = ob.intersects(ray)
      if distance is not None and distance < min_distance:
        min_distance = distance
        object_hit = ob
    return min_distance, object_hit


# Colors
LIGHT_GREEN = Vector(0.2, 0.8, 0.1)
LIGHT_BLUE = Vector(0.6, 0.8, 1)
LIGHT_RED = Vector(1, 0.2, 0.3)
LIGHT_GRAY = Vector(0.8, 0.8, 0.9)
SKY_BLUE = Vector(0.1, 0.3, 0.9)
ROYAL_BLUE = Vector(0.5, 0.3, 0.9)
RED = Vector(1, 0.3, 0.1)
YELLOW = Vector(0.8, 1, 0)
EMERALD_GREEN = Vector(0.1, 0.9, 0.1)
BLACK = Vector(0, 0, 0)
WHITE = Vector(1, 1, 1)
WHITE_BG = [255, 255, 255]
clock = pygame.time.Clock()

FRAME_COUNT = 0
FRAME_RATE = 60

def main():

  camera = Camera(640, 360, 500)
  image = Image(camera)
  objects = [Sphere(Vector(-1.2*(10**8), -1*(10**7), 4*(10**8)), 6*(10**7), Diffuse(WHITE, False)),
             Sphere(Vector(1.3*(10**8), 5*(10**7), 4*(10**8)), 5*(10**7), Diffuse(WHITE, False)),
             Sphere(Vector(0, 0, 1400000000), 10 ** 9, Diffuse(LIGHT_BLUE, False)),
             Sphere(Vector(0, 0, -1400000000), 10 ** 9, Diffuse(LIGHT_BLUE, False)),
             Sphere(Vector(-1220000000, 0, 5*(10**8)), 10**9, Diffuse(LIGHT_RED, False)),
             Sphere(Vector(1220000000, 0, 5*(10**8)), 10**9, Diffuse(LIGHT_GREEN, False)),
             Sphere(Vector(0, -1045000000, 5*(10**8)), 10 ** 9, Diffuse(LIGHT_GRAY, False)),
             Sphere(Vector(0, 1120000000, 5*(10**8)), 10**9, Diffuse(WHITE, True))]
  engine = Engine(image, camera, objects)

  # # clock for testing
  # font = pygame.font.Font(None, 25)
  # total_seconds = FRAME_COUNT // FRAME_RATE
  # minutes = total_seconds // 60
  # seconds = total_seconds % 60
  # output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)
  # text = font.render(output_string, True, WHITE_BG)
  # image.window.blit(text, [250, 250])

  run = True
  runs_counter = 0
  while run:
    runs_counter += 1
    engine.render()
    print("End of rendering:", runs_counter)
    wait = True
    if runs_counter >= 25:
      print("Final Image")
      while wait:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            wait = False
            run = False
          elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            wait = False
    clock.tick(FRAME_RATE)
  pygame.quit()

if __name__ == '__main__':
  main()


# Cornell Box example
# Aliasing
# pypy, pygame cffi 0.2.1 github, Nuitka

