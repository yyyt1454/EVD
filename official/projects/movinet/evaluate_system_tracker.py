import numpy as np
import cv2
#import matplotlib.pyplot as plt
#import mediapy as media
from dataloader_tf import * 
from model import *
import tensorflow as tf
import tensorflow_hub as hub

import sys 
sys.path.append('/EVD/')
#sys.path.append('/home/ahreumseo/research/violence/datasets/MoViNet-TF/EVD/')
from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model

#from model_profiler import model_profiler
import time

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])



## CCTV busan
#data_loader = Dataloader_CCTV(img_file_path = '/home/ahreumseo/research/violence/datasets/CCTV_busan/data_npy/512'       , label_file = '/home/ahreumseo/research/violence/datasets/CCTV_busan/label/label_motionobj.csv', target_module = 'object')

## RWF-2000
# data_path = '/home/ahreumseo/research/violence/datasets/RWF2000-Video-Database-for-Violence-Detection/Dataset/dataset_npy_5fps_512/val'
# data_loader = DataGenerator_past(directory=data_path,
#                               batch_size=1, 
#                               data_augmentation=False,
#                              shuffle=False,
#                              init_states = None,
#                              train = False)

def motion_detector(curr_frame, prev_edge_intensity, threshold1, threshold2, round_param=-3):
    curr_frame = cv2.GaussianBlur(curr_frame, (5, 5), 0.3)
    curr_edge_intensity = cv2.Canny(curr_frame, threshold1, threshold2)
    curr_edge_intensity = round(np.sum(curr_edge_intensity), round_param)
    
    if curr_edge_intensity > prev_edge_intensity:
        motion_status = 'new_human'
    elif curr_edge_intensity < prev_edge_intensity:
        motion_status = 'missed'
    else:
        motion_status = 'stable'
        
    return motion_status, curr_edge_intensity

start = time.time()
model_path = '/EVD/official/projects/movinet/models/efficientdet_d0_coco17_tpu-32/saved_model'
od_model = tf.saved_model.load(model_path)
#od_model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")
end = time.time()
print(f'od model loaded : {end-start}') 

for i in range(10):
    start = time.time()
    temp_image = tf.zeros([1,512,512,3],dtype=tf.uint8)
    od_model(temp_image)
    end = time.time()
    print(f'od model inferenced: {end-start}') 





def object_detector(frame, od_model, thresh) -> pd.DataFrame:
    df = {'ymin':[], 'xmin':[], 'ymax':[], 'xmax':[]}
    result = od_model(frame[np.newaxis])
    final_box = [box for box, score, cls in zip(result['detection_boxes'][0], result['detection_scores'][0], result['detection_classes'][0])\
                 if (score >= thresh) and (cls==1)]
    im_size = frame.shape[0]
    
    for ymin, xmin, ymax, xmax in final_box:
        df['ymin'].append(ymin.numpy()*im_size)
        df['xmin'].append(xmin.numpy()*im_size)
        df['ymax'].append(ymax.numpy()*im_size)
        df['xmax'].append(xmax.numpy()*im_size)

    return pd.DataFrame(df)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = (0,255,255) 
# Initiate ORB detector
orb = cv2.ORB_create() # orb 객체 생성

def object_tracker_orb(prev_frame, prev_kpoint, curr_frame, lk_params, same_count):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    curr_kpoint, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_kpoint, None, **lk_params)
    
    # Tracked points
    good_old = prev_kpoint
    good_new = curr_kpoint

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        a,b = int(a), int(b)
        c,d = old.ravel()
        c,d = int(c), int(d)
        if (a,b) == (c,d):
            same_count += 1

    # Now update the previous frame and previous points
    # if same_count >= 5:
    #     curr_kpoint = None
    #     same_count = 0
    # else:
    #     curr_kpoint = good_new.reshape(-1,1,2)
    curr_kpoint = good_new.reshape(-1,1,2)
  
    return curr_kpoint, None, curr_frame, same_count

def keypoint_extraction_orb(frame, od_result, orb):
    
    # od 수행 
    bb = od_result[['xmin', 'ymin', 'xmax', 'ymax']]

    # 키포인트 추출
    kp, des = orb.detectAndCompute(frame,None)
    # 키포인트 정렬 
    kp.sort(key=lambda x: x.response, reverse = True)
    # 좌표로 변환
    kp_npy = np.array([np.array(k.pt) for k in kp])
    

    kpoint = []
    # 키포인트 중 바운딩 박스 내의 최대 스코어 찾기 
    for row in bb.itertuples():
        ll = np.array([row.xmin, row.ymin])
        ur = np.array([row.xmax, row.ymax])
        inidx = np.all(np.logical_and(ll <= kp_npy, kp_npy <= ur), axis=1)  
        # 최댓값만 저장 
        if kp_npy[inidx].shape[0] > 0:
            kpoint.append(kp_npy[inidx][0])
        # 해당하는 점이 없을 경우 bb의 센터포인트 저장 
        else:
            kpoint.append(np.array([(row.xmin+row.xmax)/2, (row.ymin+row.ymax)/2]))       
        
    return np.array(kpoint, dtype=np.float32).reshape(-1,1,2)

            
            
## Violence detector 
batch_size = 1
num_frames = 25
resolution = 512
checkpoint_dir = "/EVD/offical/projects/movinet/models/movinets"
model_id='a0'


backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=False,
    use_external_states=True,
)

model = build_classifier(backbone, num_classes=2, batch_size=batch_size, num_frames=num_frames, resolution=resolution, freeze_backbone=False)
init_states = model.init_states([batch_size, num_frames, resolution, resolution, 3])

image_input = tf.keras.layers.Input(shape=[None, None, None, 3],
                                        dtype=tf.float32,
                                        name='image')

state_shapes = {
    name: ([s if s > 0 else None for s in state.shape], state.dtype)
    for name, state in init_states.items()
}

states_input = {
    name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
    for name, (shape, dtype) in state_shapes.items()
}

inputs = {**states_input, 'image': image_input}
outputs = model(inputs)
    
violence_detector = CustomModel_modified(inputs, outputs, name='movinet')
violence_detector.load_weights(checkpoint_dir)

metrics = tf.keras.metrics.CategoricalAccuracy()


for j in range(1):
    
    ## CCTV busan
    #video = data_loader[j][0].copy()
    video = tf.zeros([5,512,512,3], dtype=tf.uint8)
    label = tf.constant([[0,1]], dtype = np.float32)

    ## RWF-2000
    # video, label = data_loader[j]
    # video = video[0]

    # initialization
    states = init_states
    pred = tf.constant([[0,1]], dtype = np.float32)
    execution_manager = {'detector':False, 'tracker':False}
    execution_flag = {'detector':False, 'tracker':False, 'violence':False}
    same_count=0

    for i in range(1, video.shape[0]+1, 1):
        total += 1
        

        # people count array: 5프레임동안 사람이 있었는지, 없었는지 저장하는 array 
        if (i-1) % 5 == 0:
            people_count_array = []


        # 첫 프레임: object detector 실행 후 현재 몇 명 있는지 확인
        # 초기 edge intensity 값 설정 / bounding box 정보 tracker로 넘겨주기 
        # violence detector는 실행 x
        if (i-1) == 0: 
            start = time.time()
            _, edge_intensity = motion_detector(video[i-1], 0, 200, 200, -6)
            end = time.time()
            # print(f'motion detector: {end-start} sec')

            start = time.time()
            od_result = object_detector(video[i-1], od_model, 0.2)
            people_count = len(od_result)
            people_count_array.append(bool(people_count>=2))
            od += 1
            execution_flag['detector'] = True
            end = time.time()
            print(f'object detector: {end-start} sec')

            # 만약 사람이 없다 -> motion detector만 계속 실행
            if people_count == 0:
                pass
            # 사람이 있다 -> tracker 키고, keypoint 정보 저장 
            else:
                execution_manager = {'detector':False, 'tracker':True}
                kpoint = keypoint_extraction_orb(video[i-1], od_result, orb)
            continue

        # 2프레임 ~: 
        motion_status, edge_intensity = motion_detector(video[i-1], edge_intensity, 200, 200, -6)

        if (motion_status == 'stable') and (people_count==0):
            if (not execution_flag['tracker']) and (not execution_flag['detector']):
                execution_manager = {'detector':False, 'tracker':False}
                people_count_array.append(bool(people_count>=1))

            elif execution_flag['detector']:
                execution_manager = {'detector':False, 'tracker':False}
                people_count_array.append(bool(people_count>=1))

            elif execution_flag['tracker']:
                execution_manager = {'detector':True, 'tracker':False}
            
        elif (motion_status == 'stable') and (people_count>0):
            if execution_flag['detector']:
                execution_manager = {'detector':False, 'tracker':True}

            elif execution_flag['tracker']:
                execution_manager = {'detector':False, 'tracker':True}       

        else:
            execution_manager = {'detector':True, 'tracker':False}

        # detector 실행 조건: 이전에는 사람이 없었지만 motion이 감지됨 or tracking하던 물체들이 모두 사라짐 (확실히 사람이 있는지 없는지 verification)
        if execution_manager['detector']:
            start = time.time()
            od_result = object_detector(video[i-1], od_model, 0.2)
            people_count = len(od_result)
            people_count_array.append(bool(people_count>=1))
            
            od += 1
            execution_flag['detector'] = True
            execution_flag['tracker'] = False
            end = time.time()
            print(f'object detector: {end-start} sec')

            start = time.time()
            # 만약 사람이 없다 -> motion detector만 계속 실행
            if people_count == 0:
                pass
            # 사람이 있다 -> keypoint 정보 저장 
            else:
                kpoint = keypoint_extraction_orb(video[i-1], od_result, orb)
            end = time.time()
            print(f'keypoint extraction: {end-start} sec')
            

        # tracker 실행 조건: 이전에 detector를 실행했었거나, tracking하던 물체가 남아있음. 
        elif execution_manager['tracker']:
            start = time.time()
            kpoint, track_status, video[i-1], same_count = object_tracker_orb(video[i-2], kpoint, video[i-1], lk_params, same_count)
            try:
                people_count = len(kpoint)
            except:
                people_count = 0  # 키포인트가 계속 멈춰있는 경우 (kpoint == None)
            people_count_array.append(bool(people_count>=1))
            ot += 1
            execution_flag['detector'] = False
            execution_flag['tracker'] = True
            end = time.time()
            # print(f'object tracker: {end-start} sec')

        # 둘 다 실행 안되었을 때: execution flag 초기화 
        else:
            execution_flag['detector'] = False
            execution_flag['tracker'] = False



        

        # 5프레임에 한 번씩 violence detector 실행시킬지 판단. (frame interval: hyperparameter)
        # 사람 있는 상태가 지속되었으면 실행 / 아니면 실행 x 

        if (i % 5 == 0) and (np.sum(np.array(people_count_array))>=3):
        # if (i%5 ==0):
            start = time.time()
            clip = preprocess_movinet(video[np.newaxis,...])[:, i-5:i, ...]
            pred, states = violence_detector({**states, 'image': clip}, training=False)
            execution_flag['violence'] = True
            vd += 1
            end = time.time()
            # print(f'violence detector: {end-start} sec')
            # pred2, states2 = violence_detector({**states2, 'image': clip}, training=False)
            # print(f'{j}th video, {i}th frame: human, people count: {people_count_array}, label: {label}, vd prediction before pre-screen: {pred}, after pre-screen: {pred2}')
            # print(f'')
            

        # violence detector가 실행이 안된 채로 false negative 발생하는 경우 세기 
        else:
            pass

        

        if (i == num_frames) and (execution_flag['violence']== False) and (np.argmax(pred) != np.argmax(label)):
            fn += 1


    metrics.update_state(label, pred)
    print(f'{j}th video is done, Accuracy so far: {metrics.result().numpy()}')
    
print(f'np sum: 3')
print(f'False negative due to pre-screening module: {fn}/{len(data_loader)}')
print(f'Number of frames processed by each module: object detector - {od}, object tracker - {ot}, violence detector - {vd}, total frame - {total}')

            
        
