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

def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))
   
# deprecated 
# label, prediction format
# x,y,w,h,angle_x,angle_y
def _iou_2d_box(self, label, prediction):
    correction = 0.01
    number_of_sampels=100
    
    offset = np.subtract(label[0:2],prediction[0:2])
    if np.linalg.norm(offset) > np.linalg.norm(prediction[2:4])+np.linalg.norm(label[2:4]):
        return 0
    
    x = np.linspace(-label[2]/2+correction, +label[2]/2-correction, np.sqrt(number_of_sampels)*label[2]/label[3])
    y = np.linspace(-label[3]/2+correction, +label[3]/2-correction, np.sqrt(number_of_sampels)*label[3]/label[2])
    
    label[4:6] = label[4:6]/np.linalg.norm(label[4:6])
    prediction[4:6] = prediction[4:6]/np.linalg.norm(prediction[4:6])
    rotation_matrix_label = np.array([[  label[4], label[5] ], 
                                      [ -label[5], label[4] ]])
    rotation_matrix_pred  = np.array([[  prediction[4], prediction[5] ], 
                                      [ -prediction[5], prediction[4] ]])
    
    
    # generate sample points
    sampels = np.meshgrid(y, x)
    sampels[0] = sampels[0].flatten()
    sampels[1] = sampels[1].flatten()
    sampels = np.array(sampels).T
    number_of_sampels = len(sampels)

    #rotate sample points 
    sampels = np.dot(sampels,rotation_matrix_label.T)
    
    intersection_counter = 0
    for point in sampels:
        # offset point center
        point = np.add(point,offset)
        # rotate point line up with prediction
        point = np.dot(point,rotation_matrix_pred)
        # rotate by 90 because...
        point = np.dot(point,np.array([[0,-1],[1,0]])) 
        
        if (    point[0] <= +prediction[2]/2
            and point[0] >= -prediction[2]/2 
            and point[1] <= +prediction[3]/2
            and point[1] >= -prediction[3]/2 ):
            intersection_counter += 1
            
    size_pred = prediction[2]*prediction[3]
    size_label = label[2]*label[3]
    
    # intersection area
    intesection_area_size = size_label*intersection_counter/number_of_sampels
    union_area_size = size_pred+size_label-intesection_area_size
    
    return intesection_area_size/union_area_size
    
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
            union = r1_area+r2_area-intersection
            return intersection/union
    else:
        return 0
