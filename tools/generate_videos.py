import os
import cv2


#  write videos
root = '/data/sdv2/workspace/tianchi/dataset/tianchiyusai/'
img_root = os.path.join(root, 'JPEGImages')
ann_root = '/data/sdv2/STM_test/test/2007091020/tianchi'
save_dir = '/data/sdv2/STM_test/test/2007091020/videos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, video_id in enumerate(os.listdir(ann_root)):
    print(i, video_id)
    ann_dir = os.path.join(ann_root, video_id)
    img_dir = os.path.join(img_root, video_id)
    namelist = sorted(os.listdir(ann_dir))

    h, w, _ = cv2.imread(os.path.join(ann_dir, namelist[0])).shape
    video_dir = os.path.join(save_dir, video_id + '.avi')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, 24, (w, h))

    for name in namelist:
        mask = cv2.imread(os.path.join(ann_dir, name))
        img = cv2.imread(os.path.join(img_dir, name.replace('png', 'jpg')))
        img_show = (img * 0.5 + mask * 0.5).astype('uint8')
        videoWriter.write(img_show)