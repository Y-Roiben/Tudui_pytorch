from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

img = Image.open(r'C:\Users\hp\Desktop\程序\DEMO\dataset\dog.jpg')
# print(img)
#
# # ToTensor
#
# trans_totensor = transforms.ToTensor()
# img_tensor = trans_totensor(img)
# print("----------totensor-----------------")
# print(img_tensor)
# print(img_tensor.size())
#
# # 归一化 Normalize
#
# print('-----------Normalize------------')
# tran_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# img_norm = tran_norm(img_tensor)
# print(img_norm)
#
# # 修改size -- Resize
# # transforms.Resize([h, w]) 同时指定长宽
# # transforms.Resize(x) 将图片短边缩放至x，长宽比保持不变：
# print("------------Resize--------------")
# print("img的size:", img.size)
# tran_resize = transforms.Resize(500)
# img_resize = tran_resize(img)
# print("img修改后的size:", img_resize.size)
#
# img_resize_totensor = trans_totensor(img_resize)
# print(img_resize_totensor)
# print(img_resize_totensor.size())

# compose 将几个变换组合在一起
# transforms.Compose([transforms.Resize([h, w]), transforms.ToTensor()])


img1 = transforms.RandomCrop((128, 128), padding=16)(img)
img2 = transforms.RandomCrop((500, 128), padding=25, fill=125, padding_mode="symmetric")(img)
axs = plt.figure().subplots(1, 3)
axs[0].imshow(img)
axs[0].set_title('src')
axs[0].axis('off')
axs[1].imshow(img1)
axs[1].set_title('RandomCrop')
axs[1].axis('off')
axs[2].imshow(img2)
axs[2].set_title('padding_mode')
axs[2].axis('off')
plt.show()
