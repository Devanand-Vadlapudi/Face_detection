import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

'''xml file contains the trained model that can identify the face'''

img= cv2.imread("photo.jpg")

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,
minNeighbors=10)

'''faces would contain all the co-ordinates of the faces
 detected and scaleFactor is like first it checks in a
 small block for faces and would increase this block
 size by 5% in the next iteration until whole image is 
 iterated so that this block would consist the required
 faces and minNeighbors is the neighbour blocks considered
 so that all together form a face'''

'''used 1.1,10 but 1.05,5 are optimal values preferred'''

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


'''x,y,w,h would be the co-ordinates required to find faces and 0,255,0
for green and 3 would be rectangle's border width'''

resized=cv2.resize(img,(int(img.shape[0]/3),int(img.shape[1]/3)))

print(faces)

cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()