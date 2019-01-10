import numpy as np
from math import pi, cos, sin

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x

    def _print(self):
        print("[%f, %f],"%(self.x,self.y))

class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        x = (self.b*other.c - self.c*other.b)/w
        y = (self.c*other.a - self.a*other.c)/w
        return Vector(x, y)

    def _print(self):
        print("[%f, %f, %f],"%(self.a,self.b,self.c))


def rectangle_vertices(cx, cy, w, h, r):
    angle = r
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def __print(rect, name):
    print("%s =  np.array(["%name)
    for r in rect:
        r._print()
    print("])")

def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1
    i = 0
    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):

        #__print(intersection,"intersection")

        if len(intersection) <= 2:
            break # No intersection
        i+=1
        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        o = 0
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0: # ecke innerhalb
                new_intersection.append(s)
            if s_value * t_value < 0: # kante schneidet innerhalb
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)
            o+=1
        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0
    return 0.5 * sum(p.cross(q) for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


# r1 = [x,y,dim_x,dim_y,angle]
# angle is in rad
def iou_pure_python(r1,r2):
    distance = np.linalg.norm( np.array([r1[0] - r2[0], r1[1] - r2[1]]))
    diagonal1 = np.linalg.norm( np.array(r1[2:4]) )
    diagonal2 = np.linalg.norm( np.array(r2[2:4]) )
    if distance < diagonal1+diagonal2:
        intersection = intersection_area(r1, r2)
        if intersection == 0:
            return 0
        else:
            r1_area = r1[2]*r1[3]
            r2_area = r2[2]*r2[3]
            union = r1_area+r2_area
            return intersection/(union - intersection)
    else:
        return 0
