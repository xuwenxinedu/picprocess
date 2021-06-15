import numpy as np 
from matplotlib import pyplot as plt 
a = plt.imread('psc.jpg')


print(a.shape)

# plt.imshow(a)		#画出该图
# plt.show()		#显示你画了的东西

# x = np.full(shape = (100,200,3),fill_value = 129)	#最后一个参数 代表色号
# x[50:,:] = [228,251,142]		#自己理解，两个冒号行列穷举了
# plt.imshow(x)
# plt.show()

#转换为灰度图

#0~255  逐渐由暗色变为亮色  0为黑   255为白
#灰度图转换 ： l = r * 0.299 + g * 0.587 + b * 0.114  rgb为最低维度的数据

# b = np.array([0.299,0.587,0.144]) 
# x = np.dot(a,b)
# plt.imshow(x,cmap = 'gray')		#第二个参数必须加 
# plt.show()

#图像颜色通道

# t = a.copy()      #获取所有像素点
# t[:,:,1:3] = 0    #g,b 设置成0	前两个冒号遍历全图，后边索引为1，2的

# red=a.copy()
# green=a.copy()
# blue=a.copy()
# red[:,:,1:3]=0
# # green[:,:,::2]=0
# green[:,:,0] = 0
# green[:,:,2] = 0
# blue[:,:,:2]=0

# fig,ax=plt.subplots(2,2)
# fig.set_size_inches(15,15)
# ax[0,0].imshow(a)
# ax[0,1].imshow(red)
# ax[1,0].imshow(green)
# ax[1,1].imshow(blue)

# plt.show()

# #图像镜面对称
# #水平镜面
# t = a.copy()
# plt.imshow(t[:,::-1])		#纵坐标中心为轴对称反转
# plt.show()
# #垂直镜面
# t = a.copy()
# plt.imshow(t[::-1])       #横坐标中心为轴对称反转
# plt.show()

# #打马赛克
# k = np.random.randint(0,256,size = (100,100,3))
# test = a.copy()
# test[300:400,400:500] = k
# plt.imshow(test)
# plt.show()

# #交换通道
# t = a.copy()
# plt.imshow(t[:,:,[1,2,0]])	#懂得都懂
# plt.show()


# def ChangeSize(res,dstx,dsty):
# 	res = res/255
# 	(srcx,srcy,srcz) = res.shape
# 	end = np.zeros(shape = (dstx,dsty,3))
# 	for x in range(0,dstx):
# 		for y in range(0,dsty):
# 			sx = (x + 0.5) * srcx / dstx - 0.5
# 			sy = (y + 0.5) * srcy / dsty - 0.5
# 			u = sx - int(sx)
# 			v = sy - int(sy)
# 			if sx >= srcx-1:
# 				sx = srcx-2
# 			if sy >= srcy-1:
# 				sy = srcy-2
# 			end[x,y] = (1-u) * (1-v) * res[int(sx),int(sy)] + (1-u) * v * res[int(sx),int(sy)+1] + u * (1-v) * res[int(sx)+1,int(sy)] + u * v * res[int(sx)+1,int(sy)+1]
# 	return end
# fangda = ChangeSize(a,2160,3240)
# fangda = (fangda * 255).astype('uint8')
# plt.imshow(fangda)
# plt.imsave('double.jpg',fangda)
# print(fangda.dtype)

#高斯滤波
def Gauss_filter(img,size = 3,sigma = 1):
	img = img/255
	r = img.copy()
	r[:,:,1:3] = 0
	g = img.copy()
	g[:,:,::2] = 0
	b = img.copy()
	b[:,:,:2] = 0
	(x_size,y_size,z_size) = img.shape
	ending = np.zeros(shape = (x_size,y_size,3))
	maxn = size // 2
	idx = np.linspace(-maxn,maxn,size)
	X,Y = np.meshgrid(idx,idx)
	gauss = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
	gauss /= np.sum(np.sum(gauss))				#加权
	# print(gauss)
	# print(img[10,10])
	# print(img[10-maxn:10+maxn+1,20-maxn:20+maxn+1,0])
	for i in np.arange(x_size):
		for j in np.arange(y_size):
			if i - maxn < 0 and j - maxn >= 0 and j + maxn < y_size:
				rt = np.vdot(r[i:i+maxn+1,j-maxn:j+maxn+1,0],gauss[:,size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j - maxn < 0 :
				rt = np.vdot(r[i:i+maxn+1,j:j+maxn+1,0],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j - maxn < 0 and i - maxn >= 0 and i + maxn < x_size:
				rt = np.vdot(r[i-maxn:i+maxn+1,j:j+maxn+1,0],gauss[size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j - maxn < 0:
				rt = np.vdot(r[i - maxn:i+1,j:j + maxn+1,0],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn < y_size and j - maxn >= 0:
				rt = np.vdot(r[i - maxn:i+1,j - maxn:j + maxn+1,0],gauss[:,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn >= y_size:
				rt = np.vdot(r[i - maxn:i+1,j - maxn:j+1,0],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j + maxn >= y_size and i - maxn >=0 and i + maxn < x_size:
				rt = np.vdot(r[i - maxn:i + maxn+1,j - maxn:j+1,0],gauss[size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j + maxn >= y_size:
				rt = np.vdot(r[i:i+maxn+1,j-maxn:j+1,0],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			else:
				rt = np.vdot(r[i-maxn:i+maxn+1,j-maxn:j+maxn+1,0],gauss)
			#img[i-maxn:i+maxn,j-maxn:j+maxn]  对应的周围的位置
			#ending[i,j] = np.vdot(img[i-maxn:i+maxn,j-maxn:j+maxn],gauss)
			r[i,j,0] = rt
	for i in np.arange(x_size):
		for j in np.arange(y_size):
			if i - maxn < 0 and j - maxn >= 0 and j + maxn < y_size:
				rt = np.vdot(g[i:i+maxn+1,j-maxn:j+maxn+1,1],gauss[:,size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j - maxn < 0 :
				rt = np.vdot(g[i:i+maxn+1,j:j+maxn+1,1],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j - maxn < 0 and i - maxn >= 0 and i + maxn < x_size:
				rt = np.vdot(g[i-maxn:i+maxn+1,j:j+maxn+1,1],gauss[size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j - maxn < 0:
				rt = np.vdot(g[i - maxn:i+1,j:j + maxn+1,1],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn < y_size and j - maxn >= 0:
				rt = np.vdot(g[i - maxn:i+1,j - maxn:j + maxn+1,1],gauss[:,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn >= y_size:
				rt = np.vdot(g[i - maxn:i+1,j - maxn:j+1,1],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j + maxn >= y_size and i - maxn >=0 and i + maxn < x_size:
				rt = np.vdot(g[i - maxn:i + maxn+1,j - maxn:j+1,1],gauss[size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j + maxn >= y_size:
				rt = np.vdot(g[i:i+maxn+1,j-maxn:j+1,0],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			else:
				rt = np.vdot(g[i-maxn:i+maxn+1,j-maxn:j+maxn+1,1],gauss)
			g[i,j,1] = rt
	for i in np.arange(x_size):
		for j in np.arange(y_size):
			if i - maxn < 0 and j - maxn >= 0 and j + maxn < y_size:
				rt = np.vdot(b[i:i+maxn+1,j-maxn:j+maxn+1,2],gauss[:,size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j - maxn < 0 :
				rt = np.vdot(b[i:i+maxn+1,j:j+maxn+1,2],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j - maxn < 0 and i - maxn >= 0 and i + maxn < x_size:
				rt = np.vdot(b[i-maxn:i+maxn+1,j:j+maxn+1,2],gauss[size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j - maxn < 0:
				rt = np.vdot(b[i - maxn:i+1,j:j + maxn+1,2],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn < y_size and j - maxn >= 0:
				rt = np.vdot(b[i - maxn:i+1,j - maxn:j + maxn+1,2],gauss[:,size//2:size+1])
				#rt = 1
			elif i + maxn >= x_size and j + maxn >= y_size:
				rt = np.vdot(b[i - maxn:i+1,j - maxn:j+1,2],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			elif j + maxn >= y_size and i - maxn >=0 and i + maxn < x_size:
				rt = np.vdot(b[i - maxn:i + maxn+1,j - maxn:j+1,2],gauss[size//2:size+1])
				#rt = 1
			elif i - maxn < 0 and j + maxn >= y_size:
				rt = np.vdot(b[i:i+maxn+1,j-maxn:j+1,2],gauss[size//2:size+1,size//2:size+1])
				#rt = 1
			else:
				rt = np.vdot(b[i-maxn:i+maxn+1,j-maxn:j+maxn+1,2],gauss)
			b[i,j,2] = rt	
	ending = r + g + b	
	return ending


test = Gauss_filter(a,size = 1, sigma = 1)
test = (test * 255).astype('uint8')
plt.imshow(test)
plt.show()


