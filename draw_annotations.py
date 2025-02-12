import numpy as np
import cv2 as cv
import json
import os

def read_label_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)

def get_current_annotations(annotations: list[dict], id: int):
    vessel = []
    calcinates = []
    for annotation in annotations:
        if annotation['image_id'] == id:
            segmentation = annotation['segmentation'][0]
            if annotation['category_id'] == 1:
                vessel.append(segmentation)
            else:
                calcinates.append(segmentation)
    return vessel, calcinates

def draw_on_images(image_files: list[tuple[int, str]]):
    images = []
    for id, image_file in image_files:
        image = cv.imread(image_file)
        print(id)
        
        
        vessel, calcinates = get_current_annotations(annotations, id)
        draw_annotation(image, vessel)
        draw_annotation(image, calcinates, (0, 0, 255))
        
        images.append(image)
    return images[:-1]

def draw_annotation(image: np.ndarray, annotation: list, color: tuple = (255, 0, 0), thickness: int = 1):
    for element in annotation:
        x_coords = element[::2]
        y_coords = element[1::2]
        coords = np.array(list(zip(x_coords, y_coords)), np.int32).reshape((-1, 1, 2))
        cv.polylines(image, [coords], isClosed=True, color=color, thickness=thickness)

if __name__ == '__main__':
    print('\n \n ')
    
    input_folder = '_00084'
    
    entry_folder = f'{os.getcwd()}\\..\\annotations\\{input_folder}'
    for folder in os.listdir(entry_folder):
        selected_folder = f'{entry_folder}\\{folder}'
        labels = read_label_json(f'{selected_folder}\\labels.json')
        
        image_files = [(i['id'], f'{selected_folder}\\{i['file_name']}') for i in labels['images']]
        categories = [i['name'] for i in labels['categories']]
        annotations = labels['annotations']
        
        images = draw_on_images(image_files)
        counter = 0
        while True:
            cv.imshow(folder, images[counter % len(images)])
            if cv.waitKey(1000) == ord('q'):
                cv.destroyAllWindows()
                break
            counter += 1
        