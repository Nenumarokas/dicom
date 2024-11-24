import numpy as np

class Point:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates: np.ndarray = coordinates
        self.nearby: list[Point] = []
        self.value: int = 0
        self.angle: np.ndarray = np.array([-1, -1, -1])
        self.cross: bool = False
        self.end: bool = False

    def add_nearby(self, point: 'Point') -> None:
        self.nearby.append(point)

    def check_state(self) -> None:
        self.cross = len(self.nearby) > 2
        self.end = len(self.nearby) == 1

    def get_surround_points(self, nearby_pixels: list) -> list[tuple]:
        cx, cy, cz = self.coordinates
        return [(cx+p[0], cy+p[1], cz+p[2]) for p in nearby_pixels]
    
    def is_cross_close(self, cross_distance: int, visited_points: list['Point'] = []) -> bool:
        if cross_distance == 0:
            return False
        unvisited = [p for p in self.nearby if p not in visited_points]

        visited_points.append(self)
        for point in unvisited:
            if point.cross:
                return True
            return point.is_cross_close(cross_distance-1, visited_points)
        return False

    def remove_point(self) -> None:
        for point in self.nearby:
            point.nearby.remove(self)
            point.check_state()
        self.value = -1
        self.cross = False
        self.end = False

    def distance_to_far_end(self, path: list['Point'] = [], distance: int = 0)\
            -> tuple[int, list['Point']]:
        path = path.copy()
        if self.end and distance > 0:
            return distance, path
        
        unvisited = [p for p in self.nearby if p not in path]
        if len(unvisited) == 0:
            return distance, path

        distance_paths = []
        path.append(self)
        for point in unvisited:
            distance_paths.append(point.distance_to_far_end(path, distance+1))
        distance_paths.sort(key = lambda x: x[0])        
        return distance_paths[-1]
    
    def path_to_end(self, last_point: 'Point', path: list['Point'] = [], distance: int = 0)\
            -> tuple[int, list['Point']]:
        path = path.copy()
        if self.end and self == last_point:
            return distance, path
        
        unvisited = [p for p in self.nearby if p not in path]
        if len(unvisited) == 0:
            return -1, path

        path.append(self)
        for point in unvisited:
            result = point.path_to_end(last_point=last_point, path=path, distance=distance+1)
            if result[0] > 0:
                return result
        return -1, path

    def calculate_top_vector(self) -> bool:        
        if len(self.nearby) != 2:
            self.angle = np.array([0, 0, 0])
            return False
        
        point1 = self.nearby[0].coordinates.copy()
        point2 = self.nearby[1].coordinates.copy()
        diff_vector = point1 - point2
        
        if diff_vector[0] == 0 and diff_vector[1] == 0:
            # if the plane is horizontal or impossible
            self.angle = np.array([0, 0, 0])
            return False
        
        # find most top-pointing point on a perpendicular plane
        # u = (-ac, -bc, (a^2)+(b^2))
        x_coord, y_coord, z_coord = diff_vector
        up_vector = (-x_coord*z_coord, -y_coord*z_coord, x_coord**2+y_coord**2)
        self.angle = np.array(up_vector).astype(float)
        return True
        
    def get_neighbour_angle_average(self) -> bool:
        if len(self.nearby) == 0:
            return False
        if self.is_angle_set():
            return True
        
        average = np.array([0.0, 0.0, 0.0])
        for point in self.nearby:
            if not point.is_angle_set():
                return False
            average += point.angle
        self.angle = average / len(self.nearby)
        return True
        
    def is_angle_set(self):
        return np.any(self.angle != 0) and np.any(self.angle != -1)
        
    def distance_to_point(self, another: 'Point') -> float:
        return np.linalg.norm(self.coordinates - another.coordinates)

    def __str__(self):
        return f'{self.coordinates} ({len(self.nearby)})'
    
    def __repr__(self):
        return f'{self.coordinates} ({len(self.nearby)})'

if __name__ == '__main__':
    print('\n \n ')
    point1 = Point(np.array([0, 4, 3]))
    point2 = Point(np.array([0, 1.5, 2]))
    point3 = Point(np.array([0, 1, 1]))
    
    point1.add_nearby(point2)
    point2.add_nearby(point1)
    point2.add_nearby(point3)
    point3.add_nearby(point2)
    
    skeleton = [point1, point2, point3]
    
    for point in skeleton:
        print('----')
        print(point.calculate_top_vector())
    
    print('\n')
    for point in skeleton:
        print('----')
        print(point.get_neighbour_angle_average())
    
    for point in skeleton:
        print(point.angle)