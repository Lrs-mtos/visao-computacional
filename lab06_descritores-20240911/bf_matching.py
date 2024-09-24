import cv2
import matplotlib.pyplot as plt

def match_images(im1,im2,kp1,kp2,des1,des2, n_show = 20, use_knn=False):
	# create BFMatcher object
	bf = cv2.BFMatcher()

	if not use_knn:
		# Match descriptors.
		matches = bf.match(des1, des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key=lambda x: x.distance)

		# Draw first matches.
		img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_show], None,
								   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	else:
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m, n in matches:
			if m.distance < 0.75 * n.distance:
				good.append([m])

		img_match = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:n_show], None,
									  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_match


def bf_orb(im1,im2, n_show = 20, use_knn=False):
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(im1,None)
	kp2, des2 = orb.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)


def bf_sift(im1,im2, n_show=20, use_knn=False):
	# Initiate SIFT detector
	sift = cv2.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(im1,None)
	kp2, des2 = sift.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

def bf_surf(im1,im2, n_show=20, use_knn=False):
	# Initiate SURF detector
	surf = cv2.xfeatures2d.SURF_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = surf.detectAndCompute(im1,None)
	kp2, des2 = surf.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

def bf_brief(im1,im2, n_show=20, use_knn=False):
	fast = cv2.FastFeatureDetector_create()
	brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

	# find the keypoints and descriptors with SIFT
	kp1 = fast.detect(im1, None)
	kp1, des1 = brief.compute(im1, kp1)

	kp2 = fast.detect(im2, None)	
	kp2, des2 = brief.compute(im2, kp2)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

############


img1 = cv2.imread("images/antartica.jpg")
img2 = cv2.imread("images/antartica_lata.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#SIFT
im_sift = bf_sift(img1,img2,use_knn=True)

#ORB
im_orb = bf_orb(img1,img2,use_knn=True)

#SURF
im_surf = bf_surf(img1,img2,use_knn=True)

#BRIEF
im_brief = bf_brief(img1,img2,use_knn=True)



plt.subplot(221).set_ylabel("SIFT"), plt.imshow(im_sift,'gray') #imagem original
plt.subplot(222).set_ylabel("ORB"), plt.imshow(im_orb,'gray') #imagem original
plt.subplot(223).set_ylabel("SURF"), plt.imshow(im_surf,'gray') #imagem original
plt.subplot(224).set_ylabel("BRIEF"), plt.imshow(im_brief,'gray') #imagem original


plt.show()

