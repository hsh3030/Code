# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, jsonify, url_for
from flask import current_app, send_from_directory, send_file
from werkzeug import secure_filename
import datetime
import os

from skimage import data, io
import cv2
import utils_model, utils
from utils_model import VGG_LOSS
from keras.models import load_model
from keras import backend as K


app = Flask(__name__, static_url_path = "", static_folder = "_generate_image")
model = False

# 파일 업로드 용량 제한하기 아래는 16MB가 최대 제한
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


# 이미지 크기에 따라 리사이즈 및 분할 계산에 필요한 기준값 리턴
def image_size_check(image, image_shape, edge_size):
    img_height, img_width, _ = image.shape

    w_ = divmod(img_width, image_shape[0] - (edge_size * 2))
    w_count = w_[0]
    if w_[1] > (image_shape[0] / 2):
        w_count += 1
    h_ = divmod(img_height, image_shape[1] - (edge_size * 2))
    h_count = h_[0]
    if h_[1] > (image_shape[1] / 2):
        h_count += 1

    # 이미지가 너무 작아서 0인 경우 기본값인 1로 변경
    if w_count == 0:
        w_count = 1
    if h_count == 0:
        h_count = 1

    return image.shape, w_count, h_count


# 현재 지정된 shape에 맞게 이미지를 분할해서 리스트로 리턴함
# 가로 방향으로 먼저 분할
def image_split(image, image_shape):
    height = image_shape[1]
    width = image_shape[0]
    
    img_height, img_width, _ = image.shape

    images = []
    for h in range(0, img_height, height):
        for w in range(0, img_width, width):
            slice_image = image[h:h + height, w:w + width, :]
            images.append(slice_image)

    return images


# 이미지 분할할 때 가장자리 겹쳐지게 분할
def image_split_overlap(image, image_shape, edge_size):
    height = image_shape[1] - (edge_size * 2)
    width = image_shape[0] - (edge_size * 2)
    h_half_size = int(image_shape[1] / 2)
    w_half_size = int(image_shape[0] / 2)
    
    img_height, img_width, _ = image.shape

    images = []
    for h in range(h_half_size, img_height, height):
        for w in range(w_half_size, img_width, width):
            slice_image = image[h - h_half_size:h + h_half_size,
                                w - w_half_size:w + w_half_size, :]
            images.append(slice_image)

    return images


# 이미지를 겹쳐서 분할한 경우 사용되지 않는 가장자리를 제거하고 리턴
def images_crop(images, w_count, h_count, edge_size):
    # 현재 분할된 이미지의 크기 가져오기
    im_shape = images[0].shape

    # 상단/하단 데이터 가져오기
    top_images = images[:w_count]
    bottom_images = images[-w_count:]

    # 각 모서리 부분 데이터 가져오기
    # 이미지 배열을 작업할 때 세로/가로 순임을 기억할 것.
    corner_top_left = top_images[0, 0:edge_size, 0:edge_size, :]
    corner_top_right = top_images[-1, 0:edge_size, -edge_size:, :]
    corner_bottom_left = bottom_images[0, -edge_size:, 0:edge_size, :]
    corner_bottom_right = bottom_images[-1, -edge_size:, -edge_size:, :]

    # 상단 가장자리 데이터 크롭
    top_image = []
    for i in range(w_count):
        to_im = top_images[i, 0:edge_size, edge_size:im_shape[0] - edge_size, :]
        if len(top_image) == 0:
            top_image = to_im
        else:
            top_image = cv2.hconcat([top_image, to_im])

    # 하단 가장자리 데이터 크롭
    bottom_image = []
    for i in range(w_count):
        bo_im = bottom_images[i, -edge_size:, edge_size:im_shape[0] - edge_size, :]
        if len(bottom_image) == 0:
            bottom_image = bo_im
        else:
            bottom_image = cv2.hconcat([bottom_image, bo_im])

    # 좌측/우측 가장자리 데이터 크롭
    left_image = []
    right_image = []
    for i in range(0, len(images), w_count):
        le_im = images[i, edge_size:im_shape[1] - edge_size, 0:edge_size]
        if len(left_image) == 0:
            left_image = le_im
        else:
            left_image = cv2.vconcat([left_image, le_im])

        ri_im = images[i + w_count - 1, edge_size:im_shape[1] - edge_size, -edge_size:]
        if len(right_image) == 0:
            right_image = ri_im
        else:
            right_image = cv2.vconcat([right_image, ri_im])

    # 모서리 데이터는 상단/하단 데이터에 붙여두기
    top_image = cv2.hconcat([corner_top_left, top_image])
    top_image = cv2.hconcat([top_image, corner_top_right])
    bottom_image = cv2.hconcat([corner_bottom_left, bottom_image])
    bottom_image = cv2.hconcat([bottom_image, corner_bottom_right])

    edge_image = dict(top=top_image, bottom=bottom_image, left=left_image, right=right_image)

    # 크롭해서 가운데 이미지만 남기기
    new_images = []
    for i in range(len(images)):
        im = images[i, edge_size:im_shape[0] - edge_size, edge_size:im_shape[1] - edge_size, :]
        new_images.append(im)

    images = utils.hr_images(new_images)

    return images, edge_image


# 분할된 이미지 합쳐서 하나의 이미지로 리턴
# 분할된 이미지는 가로 방향이 우선이므로, 가로 방향으로 먼저 합친다
def image_merge(images, w_count, h_count, edge_size):
    # image_split_overlap 함수로 겹치게 분할 했을 때만 crop 처리
    if edge_size > 0:
        images, edge_image = images_crop(images, w_count, h_count, edge_size)

    w_images = []
    h_images = []
    for i in range(images.shape[0]):
        if len(w_images) == 0:
            w_images = images[i]
        else:
            w_images = cv2.hconcat([w_images, images[i]])
        if (i + 1) >= w_count and (i + 1) % w_count == 0:
            if len(h_images) == 0:
                h_images = w_images
            else:
                h_images = cv2.vconcat([h_images, w_images])
            w_images = []
    
    one_image = h_images

    if edge_size > 0:
        one_image = cv2.hconcat([edge_image['left'], one_image])
        one_image = cv2.hconcat([one_image, edge_image['right']])
        one_image = cv2.vconcat([edge_image['top'], one_image])
        one_image = cv2.vconcat([one_image, edge_image['bottom']])
            
    return one_image


def generate(filename):
    K.clear_session()

    image_shape = (120, 120, 3)
    edge_size = 10
    use_image_shape = (image_shape[0] - edge_size * 2, image_shape[1] - edge_size * 2, image_shape[2])

    loss = VGG_LOSS(image_shape)
    optimizer = utils_model.get_optimizer()

    image = data.imread('_uploads/' + filename) # 해당 데이터는 shape 출력 시 세로, 가로 순으로 출력된다
    
    # 기존 이미지 크기와 리사이즈 할 기준값을 가져와 크기 변경
    ori_image_shape, w_count, h_count = image_size_check(image, image_shape, edge_size)
    image = cv2.resize(image, ((use_image_shape[0] * w_count) + edge_size*2, (use_image_shape[1] * h_count) + edge_size*2), interpolation=cv2.INTER_AREA)

    # 이미지 분할해서 여러 개로 가져오기
    images = image_split_overlap(image, image_shape, edge_size)
    image = utils.hr_images(images)
    print('image shape: ', image.shape)

    image = utils.normalize(image)
    model = load_model('model/gen_model50.h5', custom_objects={'vgg_loss': loss.vgg_loss})
    print('load_model...')

    pre_img = model.predict(image)
    pre_img = utils.denormalize(pre_img)

    # 분할된 이미지 하나로 합친 후 원 사이즈로 변경
    pre_img = image_merge(pre_img, w_count, h_count, edge_size)
    pre_img = cv2.resize(pre_img, (ori_image_shape[1], ori_image_shape[0]), interpolation=cv2.INTER_AREA)
    print('pre_img shape: ', pre_img.shape)

    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('_generate_image/' + filename, pre_img)
    
    return filename


@app.route('/upload_low', methods=['GET', 'POST'])
def upload_low():
    print('upload_low....')
    if request.method == 'POST':
        print("Posted file: {}".format(request.files['file']))
        f = request.files['file']
        new_file_name = secure_filename(f.filename)

        if '.' in new_file_name:
            file_name, file_extension = os.path.splitext(new_file_name)
            file_extension = file_extension[1:]
        else:
            file_extension = new_file_name

        file_name = str(datetime.datetime.now())
        new_file_name = file_name + '.' + file_extension
        new_file_name = secure_filename(new_file_name)
        f.save('_uploads/' + new_file_name)

        # 이미지 생성하기
        file_name = generate(new_file_name)
        
        # 생성된 이미지 파일명을 리턴하면 index.html에서 해당 이미지 표시 및 다운 버튼을 생성
        return file_name


@app.route('/download_gen/<path:file_name>', methods=['GET', 'POST'])
def download_gen(file_name):
    return send_from_directory('_generate_image', file_name, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
