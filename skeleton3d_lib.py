from point3d_lib import Point
from numpy.linalg import norm
import numpy as np

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
            
            heads = new_skeleton.ends
            heads.sort(key = lambda end: norm(end.coordinates - self.head.coordinates))
            new_skeleton.head = heads[0]
            
            end_counter += 1
            
            new_skeletons.append(new_skeleton)
        return new_skeletons
    
    def calculate_normals(self) -> None:
        if len(self.points) < 10:
            print('too little points')
            return
        
        # calculate the tangents for all points
        for i in range(len(self.points)-1):
            last_point = self.points[i].coordinates
            point = self.points[i+1].coordinates
            tangent = point - last_point
            self.points[i].tangent = tangent / norm(tangent)
        self.points[-1].tangent = self.points[-2].tangent.copy()
        
        # choose a normal for the first point
        if self.points[0].tangent[2] == 1:
            self.points[0].normal = np.array([0, 1, 0])    
        else:
            self.points[0].normal = np.array([0, 0, 1])
        
        # calculate the normals for the remaining points
        for i in range(1, len(self.points)-1):
            last_normal = self.points[i-1].normal
            tangent = self.points[i].tangent
            normal = last_normal - np.dot(np.dot(last_normal, tangent), tangent)
            self.points[i].normal = normal / norm(normal)
        self.points[-1].normal = self.points[-2].normal.copy()
            
        # calculate the binomials for all points
        for point in self.points:
            binormal = np.cross(point.tangent, point.normal)
            point.binormal = binormal / norm(binormal)
        
        
        
        
        # for point in self.points:
        #     point.calculate_top_normal(self.head)
    
    def __iter__(self):
        yield from self.points
        
    def __getitem__(self, index) -> Point:
        return self.points[index]
    
    def __str__(self) -> str:
        return f'{self.name} - {len(self.points)} points'
    
    def __repr__(self) -> str:
        return f'{self.name} - {len(self.points)} points'
    
    def __len__(self) -> int:
        return len(self.points)