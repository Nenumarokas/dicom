import numpy as np
from point3d_lib import Point

class Skeleton:
    def __init__(self, name: str, id: int = -1):
        self.id: int = id
        self.name: str = name
        self.points: list[Point] = []
        self.ends: list[Point] = []
        self.head: Point = np.array([0, 0, 0])
        
    def add_points(self, point_list: np.ndarray, skeleton_mask: np.ndarray) -> None:
        if len(point_list) == 0:
            return
        
        self.points = [Point(point) for point in point_list]
        for point in self.points:
            surrounding_points = point.get_surrounding_points()
            nearby_points = [p for p in surrounding_points if skeleton_mask[p]]
            for nearby_point_coordinates in nearby_points:
                for another_point in self.points:
                    if np.array_equal(nearby_point_coordinates, another_point.coordinates):
                        point.add_nearby(another_point)
                        break      
            point.check_state()
    
    def add_branch_points(self, branch_points: list[Point]) -> None:
        if len(branch_points) == 0:
            return
        
        self.points = [Point(p.coordinates) for p in branch_points]
        
        self.points[0].nearby = [self.points[1]]
        self.points[-1].nearby = [self.points[-2]]
        for i in range(1, len(self.points) - 1):
            self.points[i].nearby = [self.points[i-1], self.points[i+1]]
        
        for point in self.points:
            point.check_state()
        
        self.ends = [p for p in self.points if p.end]
    
    def remove_close_ends(self, threshold: int = 20) -> None:
        """
        Remove skeleton ends if they are closer then \'threshold\'
        distance away to the nearest cross
        """
        removed = True
        while removed:
            removed = False
            skeleton_ends = [p for p in self.points if p.end]
            for end in skeleton_ends:
                if end.is_cross_close(threshold):
                    end.remove_point()
                    removed = True
        self.points = [p for p in self.points if p.value > -1]
        self.ends = [p for p in self.points if p.end]
        
    def find_head_point(self, selected_point: Point) -> None:
        """
        Find the closest point of a skeleton relative to another point.
        """
        closest_point = self.points[0]
        min_distance = 999999
        
        for point in self.points:
            dist = selected_point.distance_to_point(another=point)
            if dist < min_distance:
                closest_point = point
                min_distance = dist
        self.head = closest_point
        
    def create_new_path_skeletons(self, min_length: int) -> list['Skeleton']:
        new_skeletons = []
        end_counter = 1
        for end in self.ends:
            path = self.head.path_to_end(end)[1]
            if len(path) < min_length:
                continue
            
            new_skeleton = Skeleton(self.name, end_counter)
            new_skeleton.add_branch_points(path)
            end_counter += 1
            
            new_skeletons.append(new_skeleton)
        return new_skeletons
    
    def calculate_normals(self) -> None:
        for point in self.points:
            point.calculate_top_normal()
    
    def __iter__(self):
        yield from self.points
        
    def __getitem__(self, index):
        return self.points[index]
    
    def __str__(self) -> str:
        return f'{self.name} - {len(self.points)} points'
    
    def __repr__(self) -> str:
        return f'{self.name} - {len(self.points)} points'
    
    def __len__(self) -> int:
        return len(self.points)