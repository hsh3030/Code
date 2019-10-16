import os
import requests
from bs4 import BeautifulSoup
from PIL import Image

ep_url = 'https://comic.naver.com/webtoon/detail.nhn?titleId=119874&no=1199&weekday=fri'

html = requests.get(ep_url).text
soup = BeautifulSoup(html, 'html.parser')

img_names = []
for tag in soup.select('.wt_viewer img'):
    img_url = tag['src']
    img_name = os.path.basename(img_url)
    img_names.append(img_name)

    headers = {'Referer': ep_url}
    img_data = requests.get(img_url, headers=headers).content
    with open(img_name, 'wb') as f:
        f.write(img_data)

# 조각난 그림을 넣을 큰 켄버스를 만든다.
WHITE = (255, 255, 255)

max_width = 0
images_height = []
for img in img_names:
    with Image.open(img) as image_pick:
        max_width = max(max_width, image_pick.width)
        images_height.append(image_pick.height)

full_height = 0
for length in images_height:
    full_height = full_height + length
size = (max_width, full_height)
print('image size: ', end="")
print(size)
print("there are", str(len(img_names)), "images")

with Image.new('RGB', size, WHITE) as canvas:
    canvas.save('canvas.png')

# 그림ㅇㄹ 하나씩 캔버스에 넣는다
temp_height = 0
for i in range(0, len(img_names)):
    with Image.open(img_names[i]) as im1:
        with Image.open('canvas.png') as im2:

            #im1에는 조각난 그림들이 하나씩 저장된다
            #im2는 전체 캔버스가 저장된다
            im2.paste(im1, box=(0, temp_height))
            temp_height = temp_height + im1.height
            im2.save('canvas.png')
            print('img'+str(i+1)+'added')
            im2.show()