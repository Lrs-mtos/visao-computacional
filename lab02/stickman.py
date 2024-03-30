import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

#abre as imagens
im_line = cv2.imread("./imagens/line.jpg")
im_circle = cv2.imread("./imagens/circle.jpg")

assert im_line is not None, "file could not be read, check with os.path.exists()"
assert im_circle is not None, "file could not be read, check with os.path.exists()"

canvas = np.zeros((300, 300, 3), dtype=np.uint8)

#dimensoes
width = im_line.shape[1] 
height = im_line.shape[0]
#matrizes de transformacao 
x_center = width/2
y_center = height/2
#rotações
M_rotation_torso = cv2.getRotationMatrix2D((x_center,y_center),90,1) # torse
M_rotation_arm = cv2.getRotationMatrix2D((x_center,y_center),180,1) # torse
M_rotation_leg = cv2.getRotationMatrix2D((x_center,y_center),125,1) # torse


# Define sizes for body parts
head = im_circle
torso = cv2.warpAffine(im_line,M_rotation_torso,(width,height))
arm = cv2.warpAffine(im_line,M_rotation_arm,(width,height))
leg = cv2.warpAffine(im_line,M_rotation_leg,(width,height))

arm = cv2.resize(im_line, (int(im_line.shape[1] * 0.75), im_line.shape[0]))
leg = cv2.resize(im_line, (int(2*(im_line.shape[1] * 0.75)), im_line.shape[0]))

#get arm size



# Calculate the positions for the body parts
head_center = (150 - head.shape[1] // 2, 50)
torso_top_center = (150 - torso.shape[1] // 2, head_center[1] + head.shape[0])
arm_left_top = (torso_top_center[0] - arm.shape[1], torso_top_center[1] )
arm_right_top = (torso_top_center[0] + torso.shape[1], torso_top_center[1])
leg_left_top = (150 - leg.shape[1] // 2, torso_top_center[1] + torso.shape[0] - leg.shape[0] // 2)
leg_right_top = (150 - leg.shape[1] // 2, torso_top_center[1] + torso.shape[0] - leg.shape[0] // 2)


# Place head, torso, arms, and legs on the canvas
canvas[head_center[1]:head_center[1] + head.shape[0], head_center[0]:head_center[0] + head.shape[1]] = head
canvas[torso_top_center[1]:torso_top_center[1] + torso.shape[0], torso_top_center[0]:torso_top_center[0] + torso.shape[1]] = torso
canvas[arm_left_top[1]:arm_left_top[1] + arm.shape[0], arm_left_top[0]:arm_left_top[0] + arm.shape[1]] = arm
canvas[arm_right_top[1]:arm_right_top[1] + arm.shape[0], arm_right_top[0]:arm_right_top[0] + arm.shape[1]] = arm
canvas[leg_left_top[1]:leg_left_top[1] + leg.shape[0], leg_left_top[0]:leg_left_top[0] + leg.shape[1]] = leg
canvas[leg_right_top[1]:leg_right_top[1] + leg.shape[0], leg_right_top[0]:leg_right_top[0] + leg.shape[1]] = leg

# Show the stick figure
cv2.imshow('Stick Figure', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()