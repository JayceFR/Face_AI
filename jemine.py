import cv2
import mediapipe as mp
import pygame

model_path = './face_landmarker.task'
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection(0.75)



#Pygame stuff
pygame.init()
display = pygame.display.set_mode((640,480))

run = True
while run:
    display.fill((0,0,0))
    success, img = cam.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)

    image_h, image_w, image_c = img.shape

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * image_w), int(bboxC.ymin * image_h), int(bboxC.width * image_w), int(bboxC.height * image_h)
            cv2.rectangle(img, bbox, (255,0,255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    '''
    surf_img = pygame.surfarray.make_surface(img)
    surf_img_copy = surf_img.copy()
    surf_img_copy = pygame.transform.rotate(surf_img_copy, -90)
    display.blit(surf_img_copy, (0,0))'''
    #pygame.surfarray.blit_array(display, img)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False
    pygame.display.flip()


