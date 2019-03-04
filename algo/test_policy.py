from nn import cnn_lstm_policy
from gym import spaces
import numpy as np
import baselines.common.tf_util as U
import cv2

def predict_action(img, load_model_path):
    ob_space1 = spaces.Box(low=-np.inf, high=np.inf, shape = (5, ), dtype=np.float)
    ob_space2 = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    ob_space = (ob_space1, ob_space2)
    ac_space = spaces.Box(low=-np.array(np.ones(2)), high=np.array(np.ones(2)))

    pi = cnn_lstm_policy.CnnPhyLSTMPolicy("pi", ob_space, ac_space, hid_size=64, num_hid_layers = 1)

    U.initialize()
    assert load_model_path is not None
    U.load_state(load_model_path)
    
    ob1 = np.array([np.cos(1), np.sin(1), 0, 0, 0])
    ob2 = process_img(img)

    ac = pi.act(True, (ob1, ob2), pi.get_initial_state())[0]
    print(pi.get_initial_state())
    return ac


def process_img(img, gray_scale = True):
    if gray_scale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    if gray_scale:
        ret_img = np.expand_dims(ret_img, -1)
    return ret_img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)

    args = parser.parse_args()
    img = cv2.imread(args.img_path)
    ##print(img.shape)
    #img = process_img(img)
    cv2.imshow("1", img)
    #print(img.shape)
    cv2.waitKey(0)
    sess = U.single_threaded_session()
    sess.__enter__()

    ac = predict_action(img, args.load_model_path)
    print(ac)

