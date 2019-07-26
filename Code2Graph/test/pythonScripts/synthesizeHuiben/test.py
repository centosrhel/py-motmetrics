"""
geometry(background, foreground):
    rotate
    sample corner -> homography matrix
photometric:
    TODO: background depth field blur

"""
import os
import glob
import random
import math
import argparse
import numpy as np
import cv2

IM_SHOW = False
IM_SAVE = True


def resize_to_max_size(img, max_size):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    if h > w:
        f = float(max_size) / h
    else:
        f = float(max_size) / w
    return cv2.resize(img, dsize=(0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)


def truncated_normal(mu=0.0, sigma=1.0, max_range=2):
    # if isinstance(mu, list) or isinstance(mu, np.ndarray):
    #     mu = random.choice(mu)
    #     print('select mu', mu)
    while True:
        x = np.random.normal(mu, sigma)
        if -max_range * sigma < (x - mu) < max_range * sigma:
            return x


def all_points_in_image(pts, height=1.0, width=1.0, margin=0.05):
    return np.all(pts[:, 0] > margin) \
           and np.all(pts[:, 0] < width-margin) \
           and np.all(pts[:, 1] > margin) \
           and np.all(pts[:, 1] < height-margin)


def sample_homography(height=1.0, width=1.0):
    pts0 = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]).astype(np.float32)
    print('pts0', pts0, pts0.dtype)

    pts1 = pts0.copy()
    while True:
        amp_x = 0.1
        amp_y = 0.1
        disp_y = truncated_normal(0, amp_y/2)
        disp_left = truncated_normal(0, amp_x/2)
        disp_right = truncated_normal(0, amp_x/2)
        pts2 = pts1 + np.asarray([[disp_left, -disp_y],
                                  [disp_right, +disp_y],
                                  [disp_right, -disp_y],
                                  [disp_left, +disp_y]]).astype(np.float32)
        print('persp pts2', pts2, pts2.dtype)
        if True:  # all_points_in_image(pts2):
            pts1 = pts2
            break
    while True:
        scaling_amplitude = 0.1
        for mu_x, mu_y in [[0.8, 0.8], [0.8, 0.4], [0.4, 0.8],
                           [0.5, 0.5], [0.5, 0.25], [0.25, 0.5]]:
            scale_x = truncated_normal(mu_x, scaling_amplitude)
            scale_y = truncated_normal(mu_y, scaling_amplitude)
        print('scale', scale_x, scale_y)
        center = np.mean(pts1, axis=0)
        pts2 = pts1 - center
        pts2[:, 0] = pts2[:, 0] * scale_x
        pts2[:, 1] = pts2[:, 1] * scale_y
        pts2 = pts2 + center
        print('scaled pts2', pts2)
        if all_points_in_image(pts2):
            pts1 = pts2
            break
    while True:
        translation_amplitude = 0.5
        # trans_x = truncated_normal(0.0, translation_amplitude)
        # trans_y = truncated_normal(0.0, translation_amplitude)
        trans_x = np.random.uniform(-translation_amplitude, translation_amplitude)
        trans_y = np.random.uniform(-translation_amplitude, translation_amplitude)
        print('trans', trans_x, trans_y)
        pts2 = pts1 + np.asarray([trans_x, trans_y]).astype(np.float32)
        print('trans pts2')
        if all_points_in_image(pts2):
            pts1 = pts2
            break
    while True:
        max_angle = 180 / 180.0 * np.pi
        # angle = truncated_normal(0, max_angle)
        angle = np.random.uniform(-max_angle, max_angle)
        rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).astype(np.float32)
        center = np.mean(pts1, axis=0)
        pts2 = np.matmul(rot_mat, (pts1-center).transpose()).transpose() + center
        if all_points_in_image(pts2):
            pts1 = pts2
            break

    pts0[:, 0] *= width
    pts0[:, 1] *= height
    pts1[:, 0] *= width
    pts1[:, 1] *= height

    H = cv2.getPerspectiveTransform(pts0, pts1)
    print('H', H)
    return H


def warp_mask(H, height, width):
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask_warp = cv2.warpPerspective(mask, H, dsize=(width, height), flags=cv2.INTER_CUBIC)
    mask_warp = cv2.GaussianBlur(mask_warp.copy(), (3, 3), 3)
    return mask_warp


def warp_points(pts, H):
    pts1 = np.append(pts, np.ones((len(pts), 1)), axis=1)
    pts1 = pts1.transpose()
    pts2 = np.matmul(H, pts1)
    pts2 = pts2[:2, :] / pts2[2, :]
    pts2 = pts2.transpose()
    return pts2


def draw_points(img, pts):
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 10, (0, 0, 255), lineType=cv2.LINE_AA)
    return img


def additive_gaussian_noise(img, random_state=None, std=(5, 95)):
    """ Add gaussian noise to the current image pixel-wise
    Parameters:
      std: the standard deviation of the filter will be between std[0] and std[0]+std[1]
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    sigma = std[0] + random_state.rand() * std[1]
    gaussian_noise = random_state.randn(*img.shape) * sigma
    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def additive_speckle_noise(img, intensity=5):
    """ Add salt and pepper noise to an image
    Parameters:
      intensity: the higher, the more speckles there will be
    """
    # noise = np.zeros(img.shape[:2], dtype=np.float32)
    # cv2.randu(noise, 0, 1)
    noise = np.random.rand(img.shape[0], img.shape[1])
    black = noise < intensity
    white = noise > 1 - intensity
    noisy_img = img.copy()
    noisy_img[white] = 255
    noisy_img[black] = 0
    return noisy_img


def random_brightness(img, random_state=None, max_change=50):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    brightness = random_state.randint(-max_change, max_change)
    new_img = img.astype(np.int16) + brightness
    return np.clip(new_img, 0, 255).astype(np.uint8)


def random_contrast(img, random_state=None, max_change=[0.5, 1.5]):
    """ Change the contrast of img
    Parameters:
      max_change: the change in contrast will be between 1-max_change and 1+max_change
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    contrast = random_state.uniform(*max_change)
    mean = np.mean(img, axis=(0, 1))
    new_img = np.clip(mean + (img - mean) * contrast, 0, 255)
    return new_img.astype(np.uint8)


def add_shade(img, random_state=None, nb_ellipses=20,
              amplitude=[-0.5, 0.8], kernel_size_interval=(250, 350)):
    """ Overlay the image with several shades
    Parameters:
      nb_ellipses: number of shades
      amplitude: tuple containing the illumination bound (between -1 and 0) and the
        shawdow bound (between 0 and 1)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    transparency = random_state.uniform(*amplitude)

    min_dim = min(img.shape[:2]) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        angle = random_state.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float), (kernel_size, kernel_size), 0)
    mask = mask[:, :, np.newaxis]
    shaded = img * (1 - transparency * mask/255.)
    shaded = np.clip(shaded, 0, 255)
    return shaded.astype(np.uint8)


def motion_blur(img, max_ksize=10):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_ksize+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img.astype(np.uint8), -1, kernel)
    return img


def combine_fg_bg(fg, bg, mask):
    mask1 = mask / 255.0
    mask1 = mask1[:, :, np.newaxis]
    result = fg * mask1 + bg * (1.0 - mask1)
    return result.astype(np.uint8)
    # return cv2.add(fg * mask1, bg * (1.0 - mask1))


def sort_clockwise(points):
    # print('points', points)
    center = np.mean(points, axis=0)
    # print('center', center)
    point_angle = []
    for p in points:
        a = math.atan2(p[1] - center[1], p[0] - center[0])
        point_angle.append((p, a))
    return np.array([pa[0] for pa in sorted(point_angle, key=lambda x: x[1])])


def sort_gt(gt):
    '''
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt:
    :return: sorted gt
    '''
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]
    return np.asarray((tl, tr, br, bl))


def test():
    img_fn_list = glob.glob('images_000/id_*_pos_*')
    print('n_images', len(img_fn_list))
    random.shuffle(img_fn_list)
    img_fn_list = img_fn_list[:100]

    for img_fn in img_fn_list:
        print('img_fn', img_fn)
        img = cv2.imread(img_fn)
        # img = resize_to_max_size(img, 800)
        img = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
        print('img shape', img.shape)
        cv2.imshow('img', img)
        height, width = img.shape[:2]
        H = sample_homography(height, width)
        img_warp = cv2.warpPerspective(img, H, dsize=(width, height), flags=cv2.INTER_CUBIC)
        mask_warp = warp_mask(H, height, width)
        pts = np.asarray([[0, 0], [width, 0], [width, height], [0, height]]).astype(np.float32)
        pts_wrap = warp_points(pts, H)
        print('pts', pts)
        print('pts_wrap', pts_wrap)
        img_warp = draw_points(img_warp, pts_wrap)
        cv2.imshow('warp', img_warp)
        cv2.imshow('mask', mask_warp)
        cv2.waitKey(0)


def test_bg():
    bg_fn_list = glob.glob('/Users/yongjian.cyj/Downloads/dtd/images/*/*.jpg')
    print('n_bg', len(bg_fn_list))
    random.shuffle(bg_fn_list)
    bg_fn_list = bg_fn_list[:10]
    for bg_fn in bg_fn_list:
        print('bg_fn', bg_fn)
        bg = cv2.imread(bg_fn)
        bg = cv2.resize(bg, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('bg', bg)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fg', type=str, required=True)
    parser.add_argument('--bg', type=str, required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=1280)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    print('args', args)

    img_fn_list = glob.glob(args.fg)  # 'images_000/id_*_pos_*'
    print('n_images', len(img_fn_list))

    bg_fn_list = glob.glob(args.bg)  # '/Users/yongjian.cyj/Downloads/dtd/images/*/*.jpg'
    print('n_bg', len(bg_fn_list))

    result_width, result_height = args.width, args.height

    max_count = args.num
    count = 0
    while count < max_count:
        print('count', count)
        count += 1
        img_fn = random.choice(img_fn_list)
        print('img_fn', img_fn)
        img = cv2.imread(img_fn)
        # img = resize_to_max_size(img, 800)
        img = cv2.resize(img, dsize=(result_width, result_height), interpolation=cv2.INTER_CUBIC)
        print('img shape', img.shape)
        if IM_SHOW:
            cv2.imshow('img', img)
        height, width = img.shape[:2]
        H = sample_homography(height, width)
        img_warp = cv2.warpPerspective(img, H, dsize=(width, height), flags=cv2.INTER_CUBIC)
        mask_warp = warp_mask(H, height, width)
        pts = np.asarray([[0, 0], [width, 0], [width, height], [0, height]]).astype(np.float32)
        pts_wrap = warp_points(pts, H)
        print('pts', pts)
        print('pts_wrap', pts_wrap)
        if False:  # show warp corner
            img_warp = draw_points(img_warp, pts_wrap)
        if IM_SHOW:
            cv2.imshow('warp', img_warp)
            cv2.imshow('mask', mask_warp)

        bg_fn = random.choice(bg_fn_list)
        print('bg_fn', bg_fn)
        try:
            bg = cv2.imread(bg_fn)
            bg = cv2.resize(bg, dsize=(result_width, result_height), interpolation=cv2.INTER_CUBIC)
        except:
            print('read image error')
            count -= 1
            continue
        if IM_SHOW:
            cv2.imshow('bg', bg)

        result = combine_fg_bg(img_warp, bg, mask_warp)

        result = random_brightness(result, max_change=75)
        result = random_contrast(result, max_change=[0.3, 1.8])
        result = additive_speckle_noise(result, intensity=0.00035)
        result = additive_gaussian_noise(result, std=(0, 15))
        result = add_shade(result, amplitude=[-0.5, 0.8], kernel_size_interval=[50, 100])
        result = motion_blur(result, max_ksize=7)

        if IM_SHOW:
            cv2.imshow('comb', result)

        if IM_SHOW:
            cv2.waitKey(0)

        if IM_SAVE:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            result_fn = os.path.join(args.output_dir, '%08d.jpg' % count)
            cv2.imwrite(result_fn, result)
            result_csv_fn = '%s.csv' % result_fn
            with open(result_csv_fn, 'wt') as result_csv:
                pts_wrap = sort_clockwise(pts_wrap)
                pts_wrap = sort_gt(pts_wrap)
                for pt in pts_wrap:
                    result_csv.write('%f %f\n' % (pt[0], pt[1]))


if __name__ == '__main__':
    # test()
    # test_bg()
    main()
