
import imgutils as im

#X, Y,enc = im.imgdataset("./Data/Test")
X2, Y2, enc2 = im.imgdatasetreg("./chest_xray/train", ['bacteria', 'virus'], 'binary')
print(Y2)