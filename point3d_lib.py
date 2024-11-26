import numpy as np
import itertools

NEARBY_PIXEL_OFFSET = list(itertools.product([-1, 0, 1], repeat=3))
NEARBY_PIXEL_OFFSET.remove((0, 0, 0))

class Point:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates: np.ndarray = coordinates
        self.nearby: list[Point] = []
        self.value: int = 0
        self.top_normal: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.cross: bool = False
        self.end: bool = False

    def add_nearby(self, point: 'Point') -> None:
        self.nearby.append(point)

    def check_state(self) -> None:
        self.cross = len(self.nearby) > 2
        self.end = len(self.nearby) == 1

    def get_surrounding_points(self) -> list[tuple]:
        cx, cy, cz = self.coordinates
        return [(cx+p[0], cy+p[1], cz+p[2]) for p in NEARBY_PIXEL_OFFSET]
    
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

    def calculate_top_normal(self) -> bool:
        """
        p - current point coords
        p1 - neighbour 1 coords
        p2 - neighbour 2 coords
        n - plane normal
        d - where the plane intersects with the vertical origin (0, 0, z)
        v - current point coords to the plane intersection
        np - most top-facing vector on a plane of an artery cross section centered on a skeleton point
        """
        if len(self.nearby) != 2:
            return False
        
        point1 = self.nearby[0].coordinates.copy()
        point2 = self.nearby[1].coordinates.copy()
        diff_vector = (point1 - point2).astype(float)
        
        if diff_vector[0] == 0 and diff_vector[1] == 0:
            # if the plane is horizontal or non-existant
            return False
        
        # n = (p2-p1) / |p2-p1|
        plane_normal = diff_vector / np.linalg.norm(diff_vector)
        # d = |n·p| / n.z
        vertical_intersection = np.dot(plane_normal, self.coordinates) / plane_normal[2]
        # v = (0, 0, d) - p
        vertical_point = np.array([0, 0, vertical_intersection]) - self.coordinates
        vertical_point = vertical_point if vertical_intersection > 0 else -vertical_point
        # np = (v / |v|) + p
        self.top_normal = (vertical_point / np.linalg.norm(vertical_point)) + self.coordinates
            
        return True
        
    def get_neighbour_top_normal_average(self) -> bool:
        if len(self.nearby) == 0:
            return False
        if self.is_top_normal_set():
            return True
        
        average = np.array([0.0, 0.0, 0.0])
        for point in self.nearby:
            if not point.is_top_normal_set():
                return False
            average += point.top_normal - point.coordinates
        self.top_normal = average / len(self.nearby) + self.coordinates
        return True
        
    def is_top_normal_set(self):
        return np.any(self.top_normal != 0) and np.any(self.top_normal != -1)
        
    def distance_to_point(self, another: 'Point') -> float:
        return np.linalg.norm(self.coordinates - another.coordinates)

    def __str__(self):
        return f'{self.coordinates} ({len(self.nearby)})'
    
    def __repr__(self):
        return f'{self.coordinates} ({len(self.nearby)})'

if __name__ == '__main__':
    print('\n \n ')
    point1 = Point(np.array([0, 1, 2]))
    point2 = Point(np.array([1, 1, 3]))
    point3 = Point(np.array([1, 2.5, 4]))
    point4 = Point(np.array([2, 3, 4]))
    point5 = Point(np.array([3, 4, 3]))
    
    point1.add_nearby(point2)
    
    point2.add_nearby(point1)
    point2.add_nearby(point3)
    
    point3.add_nearby(point2)
    point3.add_nearby(point4)
    
    point4.add_nearby(point3)
    point4.add_nearby(point5)
    
    point5.add_nearby(point4)
    
    
    skeleton = [point1, point2, point3, point4, point5]
    
    for point in skeleton:
        print('----')
        point.calculate_top_normal()
    
    print('\n')
    for point in skeleton:
        point.get_neighbour_top_normal_average()
    
    for i, point in enumerate(skeleton):
        print(tuple([round(i, 2) for i in point.top_normal]))
