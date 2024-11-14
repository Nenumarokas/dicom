import numpy as np

class Point:
    def __init__(self, coordinates):
        self.coordinates: np.ndarray = coordinates
        self.nearby: list[Point] = []
        self.value: int = 0
        self.angle: float = 0
        self.cross: bool = False
        self.end: bool = False

    def add_nearby(self, point: 'Point') -> None:
        self.nearby.append(point)

    def check_state(self) -> None:
        self.cross = len(self.nearby) > 2
        self.end = len(self.nearby) == 1

    def get_surround_points(self, offsets) -> list[tuple]:
        cx, cy, cz = self.coordinates
        return [(cx+o[0], cy+o[1], cz+o[2]) for o in offsets]
    
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

    def remove_point(self):
        for point in self.nearby:
            point.nearby.remove(self)
            point.check_state()
        self.value = -1
        self.cross = False
        self.end = False

    def distance_to_far_end(self, path: list['Point'] = [], distance: int = 0) -> tuple[int, list['Point']]:
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

    def calculate_angle(self) -> float:
        if self.nearby != 2:
            self.angle = 0
            return -1
        left_point = self.nearby[0].coordinates.copy()
        right_point = self.nearby[1].coordinates.copy()
        self.angle = 1

    def __str__(self):
        return f'{self.coordinates} ({len(self.nearby)})'
    
    def __repr__(self):
        return f'{self.coordinates} ({len(self.nearby)})'
