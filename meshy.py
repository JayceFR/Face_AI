import cv2
import mediapipe as mp
import pygame
import bg_particles as bg_particles
import math
import numpy as np

model_path = './face_landmarker.task'
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
mpDraw = mp.solutions.drawing_utils

bg_particle_effect = bg_particles.Master()

def getRightEye(image, landmarks):
    eye_top = int(landmarks[263][1])
    eye_left = int(landmarks[362][0])
    eye_bottom = int(landmarks[374][1])
    eye_right = int(landmarks[263][0])
    return pygame.rect.Rect(eye_left, eye_top, (eye_right - eye_left), (eye_bottom - eye_top))

#Pygame stuff
pygame.init()
display = pygame.display.set_mode((640,480))

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#Irises
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]

def distance_between_points(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def blink_right(polygon_points):
    rh_right = polygon_points[RIGHT_EYE[0]]
    rh_left = polygon_points[RIGHT_EYE[8]]

    rv_top = polygon_points[RIGHT_EYE[12]]
    rv_down = polygon_points[RIGHT_EYE[4]]

    vertical_distance = distance_between_points(rv_top, rv_down)
    horizantal_distance = distance_between_points(rh_right, rh_left)

    ratio = vertical_distance/horizantal_distance

    rh_right = polygon_points[LEFT_EYE[0]]
    rh_left = polygon_points[LEFT_EYE[8]]

    rv_top = polygon_points[LEFT_EYE[12]]
    rv_down = polygon_points[LEFT_EYE[4]]

    vertical_distance = distance_between_points(rv_top, rv_down)
    horizantal_distance = distance_between_points(rh_right, rh_left)

    ratio2 = vertical_distance/horizantal_distance

    final_ratio = (ratio + ratio2)/2

    if final_ratio < 0.24:
        return True
    return False

def blink_left(polygon_points):
    rh_right = polygon_points[RIGHT_EYE[0]]
    rh_left = polygon_points[RIGHT_EYE[8]]

    rv_top = polygon_points[RIGHT_EYE[12]]
    rv_down = polygon_points[RIGHT_EYE[4]]

    vertical_distance = distance_between_points(rv_top, rv_down)
    horizantal_distance = distance_between_points(rh_right, rh_left)

    ratio = vertical_distance/horizantal_distance

    rh_right = polygon_points[LEFT_EYE[0]]
    rh_left = polygon_points[LEFT_EYE[8]]

    rv_top = polygon_points[LEFT_EYE[12]]
    rv_down = polygon_points[LEFT_EYE[4]]

    vertical_distance = distance_between_points(rv_top, rv_down)
    horizantal_distance = distance_between_points(rh_right, rh_left)

    ratio2 = vertical_distance/horizantal_distance

    final_ratio = (ratio + ratio2)/2

    if final_ratio < 0.24:
        return True
    return False
    

run = True
while run:
    time = pygame.time.get_ticks()
    display.fill((0,0,0))
    success, img = cam.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    image_h, image_w, image_c = img.shape


    polygon_points = []
    right_rect = []
    eye_rect = []
    #print(mp.solutions.face_mesh.FACEMESH_IRISES)
    for loc in mp.solutions.face_mesh.FACEMESH_CONTOURS:
        eye_rect.append(loc)

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            for eachland in faces.landmark:
                polygon_points.append(list((int(eachland.x * image_w), int(eachland.y * image_h))))
            right_rect = getRightEye(img, polygon_points)
            #pygame.draw.polygon(display, (255,0,0), faces.landmark)
            #mpDraw.draw_landmarks(img, faces)
            mpDraw.draw_landmarks(img, faces, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
    
    
    print(len(polygon_points))
    if len(polygon_points) > 0:

        bg_particle_effect.recursive_call(time, display, [0,0], 1, [polygon_points[LIPS[0]][0], polygon_points[LIPS[0]][0] + 90], [polygon_points[LIPS[0]][1] + 15, polygon_points[LIPS[0]][1] + 15])
        #print(polygon_points[LIPS[0]][1], polygon_points[LIPS[0]][1] - 14)
        pygame.draw.polygon(display, (57,9,71), polygon_points[0:len(polygon_points)//2])
        pygame.draw.polygon(display, (31,16,42), polygon_points[len(polygon_points)//2 + 1:len(polygon_points)])
        if not blink_right(polygon_points):
            pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in RIGHT_EYE])
            pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in RIGHT_EYEBROW])
        else:
            pygame.draw.polygon(display, (0,0,255), [(polygon_points[p][0] , polygon_points[p][1] + 10) for p in RIGHT_EYEBROW])
        if not blink_left(polygon_points):
            pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in LEFT_EYE])
            pygame.draw.polygon(display, (0,0,255), [(polygon_points[p][0], polygon_points[p][1]) for p in LEFT_EYEBROW])
        else:
            pygame.draw.polygon(display, (0,0,255), [(polygon_points[p][0] , polygon_points[p][1] + 10) for p in LEFT_EYEBROW])
        pygame.draw.polygon(display, (255,255,255), [polygon_points[p] for p in right_iris])
        pygame.draw.polygon(display, (255,255,255), [polygon_points[p] for p in left_iris])
        pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in LIPS])
        pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in LOWER_LIPS])
        pygame.draw.polygon(display, (0,0,255), [polygon_points[p] for p in UPPER_LIPS])
    

    #cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False
    pygame.display.flip()


