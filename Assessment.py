import numpy as np 
from matplotlib import pyplot as plt 

class PicProcessing:

	#构造函数
	def __init__(self,path):
		self.img = plt.imread(path)

	#得到灰度图
	def GetGray(self):
		b = np.array([0.299,0.587,0.144]) 	# r g b系数
		x = np.dot(self.img,b)
		return x

	#双线性插值 bigger and smaller	两个参数是目标尺寸
	def ChangeSize(self,dstx,dsty):
		# -->  0,1
		pix = self.img / 255
		(srcx,srcy,srcz) = pix.shape
		end = np.zeros(shape = (dstx,dsty,3))
		for x in range(0,dstx):
			for y in range(0,dsty):
				#优化后的插值对应点
				sx = (x + 0.5) * srcx / dstx - 0.5
				sy = (y + 0.5) * srcy / dsty - 0.5
				u = sx - int(sx)
				v = sy - int(sy)
				if sx >= srcx-1:
					sx = srcx-2
				if sy >= srcy-1:
					sy = srcy-2
				end[x,y] = (1-u) * (1-v) * pix[int(sx),int(sy)] + (1-u) * v * pix[int(sx),int(sy)+1] + u * (1-v) * pix[int(sx)+1,int(sy)] + u * v * pix[int(sx)+1,int(sy)+1]
		#--> 0,255
		end = (end * 255).astype('uint8')
		return end

	#Gaussian filtering	 只做了彩图版本  灰度图版本更简单，不考虑通道即可
	def GaussFilter(self,size = 3,sigma = 1):
		(x_size,y_size,z_size) = self.img.shape
		if min(x_size,y_size) < size:
			print('size error!')
			return self.img
		#红绿蓝通道分离，分别处理
		r = self.img.copy()/255
		r[:,:,1:3] = 0		#索引为1，2的
		g = self.img.copy()/255
		g[:,:,::2] = 0		#索引为0，2的
		b = self.img.copy()/255
		b[:,:,:2] = 0		#索引为0，1的
		maxn = size // 2
		idx = np.linspace(-maxn,maxn,size)
		X,Y = np.meshgrid(idx,idx)
		#由于还要加权，系数可以省略
		gauss = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
		#最后得到的系数矩阵
		gauss /= np.sum(np.sum(gauss))
		for i in np.arange(x_size):
			for j in np.arange(y_size):	
				#边缘处理
				if i - maxn < 0 and j - maxn >= 0 and j + maxn < y_size:
					rt = np.vdot(r[i:i+maxn+1,j-maxn:j+maxn+1,0],gauss[:,size//2:size+1]) *2
					gt = np.vdot(g[i:i+maxn+1,j-maxn:j+maxn+1,1],gauss[:,size//2:size+1]) *2
					bt = np.vdot(b[i:i+maxn+1,j-maxn:j+maxn+1,2],gauss[:,size//2:size+1]) *2
				elif i - maxn < 0 and j - maxn < 0 :
					rt = np.vdot(r[i:i+maxn+1,j:j+maxn+1,0],gauss[size//2:size+1,size//2:size+1])
					gt = np.vdot(g[i:i+maxn+1,j:j+maxn+1,1],gauss[size//2:size+1,size//2:size+1])
					bt = np.vdot(b[i:i+maxn+1,j:j+maxn+1,2],gauss[size//2:size+1,size//2:size+1])
				elif j - maxn < 0 and i - maxn >= 0 and i + maxn < x_size:
					rt = np.vdot(r[i-maxn:i+maxn+1,j:j+maxn+1,0],gauss[size//2:size+1])
					gt = np.vdot(g[i-maxn:i+maxn+1,j:j+maxn+1,1],gauss[size//2:size+1])
					bt = np.vdot(b[i-maxn:i+maxn+1,j:j+maxn+1,2],gauss[size//2:size+1])
				elif i + maxn >= x_size and j - maxn < 0:
					rt = np.vdot(r[i - maxn:i+1,j:j + maxn+1,0],gauss[size//2:size+1,0:size//2+1])
					gt = np.vdot(g[i - maxn:i+1,j:j + maxn+1,1],gauss[size//2:size+1,0:size//2+1])
					bt = np.vdot(b[i - maxn:i+1,j:j + maxn+1,2],gauss[size//2:size+1,0:size//2+1])
				elif i + maxn >= x_size and j + maxn < y_size and j - maxn >= 0:
					rt = np.vdot(r[i - maxn:i+1,j - maxn:j + maxn+1,0],gauss[:,0:size//2+1])
					gt = np.vdot(g[i - maxn:i+1,j - maxn:j + maxn+1,1],gauss[:,0:size//2+1])
					bt = np.vdot(b[i - maxn:i+1,j - maxn:j + maxn+1,2],gauss[:,0:size//2+1])
				elif i + maxn >= x_size and j + maxn >= y_size:
					rt = np.vdot(r[i - maxn:i+1,j - maxn:j+1,0],gauss[0:size//2+1,0:size//2+1])
					gt = np.vdot(g[i - maxn:i+1,j - maxn:j+1,1],gauss[0:size//2+1,0:size//2+1])
					bt = np.vdot(b[i - maxn:i+1,j - maxn:j+1,2],gauss[0:size//2+1,0:size//2+1])
				elif j + maxn >= y_size and i - maxn >=0 and i + maxn < x_size:
					rt = np.vdot(r[i - maxn:i + maxn+1,j - maxn:j+1,0],gauss[0:size//2+1])
					gt = np.vdot(g[i - maxn:i + maxn+1,j - maxn:j+1,1],gauss[0:size//2+1])
					bt = np.vdot(b[i - maxn:i + maxn+1,j - maxn:j+1,2],gauss[0:size//2+1])
				elif i - maxn < 0 and j + maxn >= y_size:
					rt = np.vdot(r[i:i+maxn+1,j-maxn:j+1,0],gauss[0:size//2+1,size//2:size+1])
					gt = np.vdot(g[i:i+maxn+1,j-maxn:j+1,1],gauss[0:size//2+1,size//2:size+1])
					bt = np.vdot(b[i:i+maxn+1,j-maxn:j+1,2],gauss[0:size//2+1,size//2:size+1])
				else:
					rt = np.vdot(r[i-maxn:i+maxn+1,j-maxn:j+maxn+1,0],gauss)
					gt = np.vdot(g[i-maxn:i+maxn+1,j-maxn:j+maxn+1,1],gauss)
					bt = np.vdot(b[i-maxn:i+maxn+1,j-maxn:j+maxn+1,2],gauss)
				r[i,j,0] = rt
				g[i,j,1] = gt
				b[i,j,2] = bt
		target = r + g + b
		target = (target * 255).astype('uint8')
		return target

temp = PicProcessing('Leonardo.jpg')
gray = temp.GetGray()
x,y,z = temp.img.shape

double = temp.ChangeSize(2*x,2*y)
half = temp.ChangeSize(x//2,y//2)
blur = temp.GaussFilter(size = 9,sigma = 1.5)

#plt.imshow(temp.img)

plt.subplot(2,3,1)
plt.imshow(temp.img)
plt.title('img')

#显示位置   画哪张图片  标题   把图像保存到当前文件夹
plt.subplot(2,3,2)
plt.imshow(gray,cmap = 'gray')
plt.title('gray')
plt.imsave('gray.jpg',gray,cmap = 'gray')

plt.subplot(2,3,3)
plt.imshow(double)
plt.title('double')
plt.imsave('double.jpg',double)

plt.subplot(2,3,4)
plt.imshow(half)
plt.title('half')
plt.imsave('half.jpg',half)

plt.subplot(2,3,5)
plt.imshow(blur)
plt.title('blur')
plt.imsave('blur.jpg',blur)

plt.show()
